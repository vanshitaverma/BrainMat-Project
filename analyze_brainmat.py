#!/usr/bin/env python3
"""
analyze_brainmat.py

Full EEG + Physiological analysis pipeline for BrainMat Muse recordings.
- Batch processes pairs of CSVs per participant: participantID_before.csv and participantID_during.csv
- Optional HSI filtering (keep rows where HSI_* in HSI_ALLOWED)
- Computes absolute & relative band powers, EEG indices, HR & motion metrics
- Computes percent changes, group statistics (paired t-test, Wilcoxon fallback), Cohen's d
- Generates static Matplotlib figures and a plain-text summary report
- Organizes outputs into results/metrics/, results/figures/, results/logs/
"""

import os
import glob
import logging
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, pearsonr

# ----------------- CONFIGURATION -----------------
FILTER_GOOD_SIGNAL = True       # If True, only rows where all HSI_* in HSI_ALLOWED are kept
HSI_ALLOWED = [1, 2]           # values to keep if filtering (1 = Good, 2 = Fair)
DATA_DIR = "./data"            # folder with input CSVs
RESULTS_DIR = "./results"      # root output folder
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

# What metrics will we visualize/stat-test by default
KEY_METRICS = ["Alpha_mean", "Theta_Beta_Ratio", "Meditation_Index", "HeartRate_mean"]

# Matplotlib / Seaborn style
sns.set(style="whitegrid", font_scale=1.0)

# ----------------- Logging -----------------
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "analysis_log.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # also print to console

# ----------------- HELPER FUNCTIONS -----------------
def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and do basic cleaning."""
    try:
        df = pd.read_csv(path)
        # replace infinities and remove fully empty rows
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        logger.info(f"Loaded {path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Failed to load {path}: {e}")
        raise

def filter_hsi(df: pd.DataFrame, allowed: List[int]) -> pd.DataFrame:
    """Keep only rows where all HSI_* channels are in allowed list.
       If HSI columns missing, returns df unchanged (and logs a warning)."""
    hsi_cols = [f"HSI_{ch}" for ch in CHANNELS]
    missing = [c for c in hsi_cols if c not in df.columns]
    if missing:
        logger.warning(f"HSI columns missing: {missing}. Skipping HSI filtering for this file.")
        return df
    mask = np.ones(len(df), dtype=bool)
    for c in hsi_cols:
        mask &= df[c].isin(allowed)
    filtered = df[mask].copy()
    logger.info(f"HSI filtering: kept {len(filtered)}/{len(df)} rows (allowed={allowed})")
    return filtered

def compute_band_means(df: pd.DataFrame) -> Dict[str, float]:
    """Compute mean absolute power per band averaged across configured channels."""
    results = {}
    for band in BANDS:
        cols = [f"{band}_{ch}" for ch in CHANNELS if f"{band}_{ch}" in df.columns]
        if not cols:
            logger.debug(f"No columns found for band {band} in this file.")
            results[f"{band}_mean"] = np.nan
        else:
            # per-row mean across channels then mean across time
            per_row_mean = df[cols].mean(axis=1, skipna=True)
            results[f"{band}_mean"] = per_row_mean.mean(skipna=True)
    return results

def compute_relative_powers(band_means: Dict[str, float]) -> Dict[str, float]:
    """Given absolute band means, compute relative power per band."""
    rel = {}
    total = 0.0
    for band in BANDS:
        val = band_means.get(f"{band}_mean", np.nan)
        if not np.isnan(val):
            total += val
    if total == 0 or np.isnan(total):
        for band in BANDS:
            rel[f"Relative_{band}"] = np.nan
        return rel
    for band in BANDS:
        val = band_means.get(f"{band}_mean", np.nan)
        rel[f"Relative_{band}"] = (val / total) if not np.isnan(val) else np.nan
    return rel

def safe_log(x: float) -> float:
    """Log with guard for non-positive inputs."""
    try:
        if x <= 0 or np.isnan(x):
            return np.nan
        return math.log(x)
    except Exception:
        return np.nan

def compute_eeg_indices(df: pd.DataFrame, band_means: Dict[str, float]) -> Dict[str, float]:
    """Compute several EEG-derived indices from band_means and channel-specific columns where needed."""
    idx = {}
    # Basic ratios (from averaged band means)
    try:
        theta = band_means.get("Theta_mean", np.nan)
        beta = band_means.get("Beta_mean", np.nan)
        alpha = band_means.get("Alpha_mean", np.nan)
        delta = band_means.get("Delta_mean", np.nan)
    except Exception:
        theta = beta = alpha = delta = np.nan

    # Theta/Beta
    idx["Theta_Beta_Ratio"] = (theta / beta) if (not np.isnan(theta) and not np.isnan(beta) and beta != 0) else np.nan
    # Alpha/Delta (Calmness)
    idx["Alpha_Delta_Ratio"] = (alpha / delta) if (not np.isnan(alpha) and not np.isnan(delta) and delta != 0) else np.nan
    # Meditation index (Theta + Alpha) / Beta
    idx["Meditation_Index"] = ((theta + alpha) / beta) if (not np.isnan(theta) and not np.isnan(alpha) and not np.isnan(beta) and beta != 0) else np.nan
    # Engagement index Beta / (Alpha + Theta)
    denom = (alpha + theta) if (not np.isnan(alpha) and not np.isnan(theta)) else np.nan
    idx["Engagement_Index"] = (beta / denom) if (not np.isnan(beta) and not np.isnan(denom) and denom != 0) else np.nan
    # Focus index (same as Theta_Beta_Ratio)
    idx["Focus_Index"] = idx["Theta_Beta_Ratio"]
    # Calmness index Alpha/Delta
    idx["Calmness_Index"] = idx["Alpha_Delta_Ratio"]

    # Frontal Alpha Asymmetry (FAA) using AF8 and AF7 (channel-specific)
    af7_col = "Alpha_AF7"
    af8_col = "Alpha_AF8"
    if af7_col in df.columns and af8_col in df.columns:
        af7_mean = df[af7_col].mean(skipna=True)
        af8_mean = df[af8_col].mean(skipna=True)
        lna = safe_log(af8_mean) - safe_log(af7_mean)
        idx["Frontal_Alpha_Asymmetry"] = lna
    else:
        idx["Frontal_Alpha_Asymmetry"] = np.nan

    return idx

def compute_motion_hr_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute accelerometer-based motion metric and heart rate summary."""
    metrics = {}
    if all(c in df.columns for c in ["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"]):
        acc_mag = np.sqrt(df["Accelerometer_X"]**2 + df["Accelerometer_Y"]**2 + df["Accelerometer_Z"]**2)
        metrics["AccMag_mean"] = acc_mag.mean(skipna=True)
        metrics["AccMag_std"] = acc_mag.std(skipna=True)
    else:
        metrics["AccMag_mean"] = np.nan
        metrics["AccMag_std"] = np.nan

    if "Heart_Rate" in df.columns:
        hr = df["Heart_Rate"].replace(0, np.nan)  # guard against zeros if not wearing
        metrics["HeartRate_mean"] = hr.mean(skipna=True)
        metrics["HeartRate_std"] = hr.std(skipna=True)
    else:
        metrics["HeartRate_mean"] = np.nan
        metrics["HeartRate_std"] = np.nan

    return metrics

def percent_change(before: float, during: float) -> float:
    """Compute percent change safely; returns nan if before is nan or zero."""
    try:
        if np.isnan(before) or np.isnan(during):
            return np.nan
        if before == 0:
            # handle zero baseline; use absolute change represented or nan
            return np.nan
        return ((during - before) / before) * 100.0
    except Exception:
        return np.nan

# ----------------- SESSION SUMMARIES -----------------
def summarize_file(file_path: str, filter_hsi_flag: bool = True, hsi_allowed: List[int] = [1,2]) -> Dict[str, float]:
    """Load a Muse CSV file and return a dictionary of computed metrics for that session."""
    df = load_csv(file_path)
    if filter_hsi_flag:
        df = filter_hsi(df, hsi_allowed)

    # If after filtering df is empty, warn and proceed with NaNs
    if df.shape[0] == 0:
        logger.warning(f"No data left after HSI filtering for file: {file_path}")
        # produce NaN-filled summary
        band_means = {f"{b}_mean": np.nan for b in BANDS}
        rel_pows = {f"Relative_{b}": np.nan for b in BANDS}
        eeg_indices = {k: np.nan for k in ["Theta_Beta_Ratio","Alpha_Delta_Ratio","Meditation_Index","Engagement_Index","Focus_Index","Calmness_Index","Frontal_Alpha_Asymmetry"]}
        motion_hr = {"AccMag_mean": np.nan, "AccMag_std": np.nan, "HeartRate_mean": np.nan, "HeartRate_std": np.nan}
    else:
        band_means = compute_band_means(df)
        rel_pows = compute_relative_powers(band_means)
        eeg_indices = compute_eeg_indices(df, band_means)
        motion_hr = compute_motion_hr_metrics(df)

    session_summary = {}
    # combine dictionaries
    session_summary.update(band_means)
    session_summary.update(rel_pows)
    session_summary.update(eeg_indices)
    session_summary.update(motion_hr)
    return session_summary

def compare_sessions(before_csv: str, during_csv: str, participant_id: str,
                     filter_hsi_flag: bool = True, hsi_allowed: List[int] = [1,2]) -> pd.DataFrame:
    """Compare the two sessions and produce a one-row dataframe summarizing differences and percent changes."""
    before_summary = summarize_file(before_csv, filter_hsi_flag, hsi_allowed)
    during_summary = summarize_file(during_csv, filter_hsi_flag, hsi_allowed)

    # assemble columns: X_before, X_during, X_change (absolute), X_pctchange
    combined = {}
    for key in set(list(before_summary.keys()) + list(during_summary.keys())):
        bval = before_summary.get(key, np.nan)
        dval = during_summary.get(key, np.nan)
        change = np.nan
        pct = np.nan
        if (not np.isnan(bval)) and (not np.isnan(dval)):
            change = dval - bval
            pct = percent_change(bval, dval)
        combined[f"{key}_before"] = bval
        combined[f"{key}_during"] = dval
        combined[f"{key}_change"] = change
        combined[f"{key}_pctchange"] = pct

    combined["Participant_ID"] = participant_id
    # return a DataFrame row for convenience
    return pd.DataFrame([combined])

# ----------------- STATISTICS UTILITIES -----------------
def cohens_d_paired(before: np.ndarray, during: np.ndarray) -> float:
    """Compute Cohen's d for paired samples: mean(diff) / std(diff)"""
    try:
        diff = during - before
        mean_diff = np.nanmean(diff)
        sd_diff = np.nanstd(diff, ddof=1)
        if sd_diff == 0 or np.isnan(sd_diff):
            return np.nan
        return mean_diff / sd_diff
    except Exception:
        return np.nan

def paired_stats(before: np.ndarray, during: np.ndarray) -> Tuple[float, float, float]:
    """
    Return (t_stat, p_value, cohen_d) for paired t-test.
    If n < 10, also compute Wilcoxon signed-rank as fallback (and include its p-value).
    """
    try:
        # drop any NaNs by aligning
        mask = ~np.isnan(before) & ~np.isnan(during)
        before_clean = before[mask]
        during_clean = during[mask]
        n = len(before_clean)
        if n <= 1:
            return (np.nan, np.nan, np.nan)
        # paired t-test
        t_stat, p_val = ttest_rel(before_clean, during_clean)
        d = cohens_d_paired(before_clean, during_clean)
        # wilcoxon fallback for small samples or if user wants - we compute if n < 10
        wil_p = np.nan
        if n < 10:
            try:
                wil_stat, wil_p = wilcoxon(before_clean, during_clean)
            except Exception:
                wil_p = np.nan
        # we will return t_stat, p_val, cohen_d (wilcoxon p can be added to summary separately if needed)
        return (t_stat, p_val, d)
    except Exception:
        return (np.nan, np.nan, np.nan)

# ----------------- VISUALIZATION -----------------
def plot_bar_paired(df_summary: pd.DataFrame, metric: str, ylabel: str, out_path: str):
    """
    Create a grouped bar chart (before vs during) per participant for metric,
    annotate with percent-change mean across participants.
    """
    # required column names
    before_col = f"{metric}_before"
    during_col = f"{metric}_during"
    pct_col = f"{metric}_pctchange"

    if before_col not in df_summary.columns or during_col not in df_summary.columns:
        logger.warning(f"Metric columns missing for plotting: {metric}")
        return

    x = df_summary["Participant_ID"].astype(str)
    before_vals = df_summary[before_col].values
    during_vals = df_summary[during_col].values
    x_pos = np.arange(len(x))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 0.7*len(x)), 5))
    ax.bar(x_pos - width/2, before_vals, width, label="Before")
    ax.bar(x_pos + width/2, during_vals, width, label="During")

    # annotate percent change per bar group above the pair with mean percent change
    mean_pct = np.nanmean(df_summary[pct_col].values) if pct_col in df_summary.columns else np.nan
    if not np.isnan(mean_pct):
        annotation = f"Mean %Δ = {mean_pct:.1f}%"
    else:
        annotation = "Mean %Δ = n/a"

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_xlabel("Participant ID")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} — Before vs During")
    ax.legend()
    # small annotation
    ax.text(0.99, 0.01, annotation, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")

def plot_paired_lines(df_summary: pd.DataFrame, metric: str, ylabel: str, out_path: str):
    """Plot paired lines (before->during) per participant for metric to show direction of change."""
    before_col = f"{metric}_before"
    during_col = f"{metric}_during"
    if before_col not in df_summary.columns or during_col not in df_summary.columns:
        logger.warning(f"Metric columns missing for plotting lines: {metric}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df_summary.iterrows():
        b = row[before_col]
        d = row[during_col]
        pid = row["Participant_ID"]
        ax.plot([0, 1], [b, d], marker='o', label=str(pid))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "During"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Paired changes — {ylabel}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")

def plot_heatmap_relative_power(all_summary_df: pd.DataFrame, out_path: str):
    """
    Create heatmap of averaged relative band powers for Before and During (rows), bands (columns).
    We'll average across participants and plot Before vs During as two rows.
    """
    rel_cols_before = [f"Relative_{b}_before" for b in BANDS]
    rel_cols_during = [f"Relative_{b}_during" for b in BANDS]
    # ensure columns exist; if not, try to fallback on just before/during keys
    # create a small dataframe
    rows = {}
    if all(col in all_summary_df.columns for col in rel_cols_before):
        rows["Before"] = all_summary_df[rel_cols_before].mean(axis=0).values
    else:
        rows["Before"] = [np.nan] * len(BANDS)
    if all(col in all_summary_df.columns for col in rel_cols_during):
        rows["During"] = all_summary_df[rel_cols_during].mean(axis=0).values
    else:
        rows["During"] = [np.nan] * len(BANDS)
    heat_df = pd.DataFrame(rows, index=BANDS).T  # rows x cols (Before/During x Bands)

    fig, ax = plt.subplots(figsize=(8, 3 + 0.6 * len(BANDS)))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("Average Relative Band Power (Before vs During)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved heatmap: {out_path}")

def plot_correlation_matrix(all_summary_df: pd.DataFrame, metrics: List[str], out_path: str):
    """
    Compute Pearson correlation matrix between percent changes of selected metrics
    and plot as heatmap.
    """
    pct_cols = [f"{m}_pctchange" for m in metrics]
    # filter only existing columns
    pct_cols = [c for c in pct_cols if c in all_summary_df.columns]
    if len(pct_cols) < 2:
        logger.warning("Not enough percent-change columns for correlation matrix.")
        return
    corr_df = all_summary_df[pct_cols].corr(method='pearson')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation matrix (percent-change)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved correlation matrix: {out_path}")

# ----------------- BATCH PROCESSING -----------------
def run_analysis(before_csv: str, during_csv: str, participant_id: str,
                 filter_hsi_flag: bool = FILTER_GOOD_SIGNAL,
                 hsi_allowed: List[int] = HSI_ALLOWED) -> pd.DataFrame:
    """Run participant-level analysis and save individual CSV into metrics/"""
    safe_mkdir(METRICS_DIR)
    try:
        df_row = compare_sessions(before_csv, during_csv, participant_id, filter_hsi_flag, hsi_allowed)
        out_path = os.path.join(METRICS_DIR, f"{participant_id}_summary.csv")
        df_row.to_csv(out_path, index=False)
        logger.info(f"Saved participant summary: {out_path}")
        return df_row
    except Exception as e:
        logger.exception(f"Failed to process participant {participant_id}: {e}")
        return pd.DataFrame()

def compute_group_statistics(master_df: pd.DataFrame) -> pd.DataFrame:
    """Compute group-level stats (mean, std, t-test, Cohen's d) for a set of metrics."""
    stats_rows = []
    # determine metric base names by looking at suffixes _before in master_df columns
    base_metrics = [c[:-7] for c in master_df.columns if c.endswith("_before")]
    # remove potential duplicates
    base_metrics = sorted(list(set(base_metrics)))

    for metric in base_metrics:
        before_col = f"{metric}_before"
        during_col = f"{metric}_during"
        change_col = f"{metric}_change"
        pct_col = f"{metric}_pctchange"

        if before_col not in master_df.columns or during_col not in master_df.columns:
            continue

        before = master_df[before_col].to_numpy(dtype=float)
        during = master_df[during_col].to_numpy(dtype=float)
        change = master_df[change_col].to_numpy(dtype=float)
        pct = master_df[pct_col].to_numpy(dtype=float) if pct_col in master_df.columns else np.array([np.nan]*len(before))

        mean_before = np.nanmean(before)
        mean_during = np.nanmean(during)
        mean_change = np.nanmean(change)
        mean_pct = np.nanmean(pct)
        std_change = np.nanstd(change, ddof=1)

        t_stat, p_val, cohend = paired_stats(before, during)
        # add Wilcoxon p-value when n < 10
        wil_p = np.nan
        try:
            mask = ~np.isnan(before) & ~np.isnan(during)
            if np.sum(mask) > 1 and np.sum(mask) < 10:
                wil_stat, wil_p = wilcoxon(before[mask], during[mask])
        except Exception:
            wil_p = np.nan

        stats_rows.append({
            "Metric": metric,
            "N": int(np.sum(~np.isnan(before) & ~np.isnan(during))),
            "Mean_Before": mean_before,
            "Mean_During": mean_during,
            "Mean_Change": mean_change,
            "Std_Change": std_change,
            "Mean_PctChange": mean_pct,
            "T_stat": t_stat,
            "P_value": p_val,
            "Wilcoxon_p": wil_p,
            "Cohens_d": cohend
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_out = os.path.join(METRICS_DIR, "Statistical_Summary.csv")
    stats_df.to_csv(stats_out, index=False)
    logger.info(f"Saved statistical summary: {stats_out}")
    return stats_df

def write_session_report(stats_df: pd.DataFrame, master_df: pd.DataFrame, out_path: str):
    """Write a short human-readable session report summarizing key findings."""
    lines = []
    lines.append("BrainMat Study — Session Report")
    lines.append("="*40)
    lines.append(f"FILTER_GOOD_SIGNAL = {FILTER_GOOD_SIGNAL} | HSI_ALLOWED = {HSI_ALLOWED}")
    lines.append("")
    if stats_df.empty:
        lines.append("No statistical summary available.")
    else:
        # iterate metrics and provide interpretation when p < 0.05
        sig_metrics = stats_df[stats_df["P_value"] < 0.05] if "P_value" in stats_df.columns else pd.DataFrame()
        lines.append("Group-level statistics (selected metrics):")
        for _, r in stats_df.iterrows():
            metric = r["Metric"]
            mean_pct = r.get("Mean_PctChange", np.nan)
            p = r.get("P_value", np.nan)
            d = r.get("Cohens_d", np.nan)
            n = int(r.get("N", 0))
            lines.append(f"- {metric}: N={n}, Mean %Δ = {mean_pct:.2f}%, p={p if not np.isnan(p) else 'n/a'}, Cohen's d = {d:.3f}")
        lines.append("")
        if not sig_metrics.empty:
            lines.append("Significant changes (p < 0.05):")
            for _, r in sig_metrics.iterrows():
                metric = r["Metric"]
                mean_pct = r.get("Mean_PctChange", np.nan)
                p = r.get("P_value", np.nan)
                lines.append(f"* {metric}: mean %Δ = {mean_pct:.2f}% (p = {p:.4f})")
            lines.append("")
            lines.append("Interpretation (automated suggestions):")
            # simple rule-based interpretation
            if any(stats_df["Metric"] == "Alpha_mean"):
                alpha_row = stats_df[stats_df["Metric"] == "Alpha_mean"]
                if not alpha_row.empty and alpha_row["P_value"].iloc[0] < 0.05 and alpha_row["Mean_PctChange"].iloc[0] > 0:
                    lines.append("- Alpha increased significantly: suggests increased calm/relaxed alertness during walking on BrainMat.")
            if any(stats_df["Metric"] == "Theta_Beta_Ratio"):
                theta_row = stats_df[stats_df["Metric"] == "Theta_Beta_Ratio"]
                if not theta_row.empty and theta_row["P_value"].iloc[0] < 0.05 and theta_row["Mean_PctChange"].iloc[0] > 0:
                    lines.append("- Theta/Beta ratio increased significantly: suggests greater meditative focus during walking.")
            if any(stats_df["Metric"] == "HeartRate_mean"):
                hr_row = stats_df[stats_df["Metric"] == "HeartRate_mean"]
                if not hr_row.empty and hr_row["P_value"].iloc[0] < 0.05 and hr_row["Mean_PctChange"].iloc[0] < 0:
                    lines.append("- Heart rate decreased significantly: suggests physiological relaxation.")
        else:
            lines.append("No metrics reached significance at p < 0.05.")
    # write file
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote session report to {out_path}")

def batch_process_all(data_dir: str = DATA_DIR):
    """Main batch driver."""
    safe_mkdir(METRICS_DIR)
    safe_mkdir(FIGURES_DIR)
    safe_mkdir(LOGS_DIR)

    files = glob.glob(os.path.join(data_dir, "*.csv"))
    participants = sorted({os.path.basename(f).split('_')[0] for f in files})
    logger.info(f"Found participants: {participants}")

    all_rows = []
    for pid in participants:
        before = os.path.join(data_dir, f"{pid}_before.csv")
        during = os.path.join(data_dir, f"{pid}_during.csv")
        if os.path.exists(before) and os.path.exists(during):
            try:
                row_df = run_analysis(before, during, pid)
                if not row_df.empty:
                    all_rows.append(row_df)
            except Exception as e:
                logger.exception(f"Error processing {pid}: {e}")
        else:
            logger.warning(f"Missing files for participant {pid} - skipping")

    if not all_rows:
        logger.error("No complete participant data processed. Exiting.")
        return

    master_df = pd.concat(all_rows, ignore_index=True)
    master_out = os.path.join(METRICS_DIR, "All_Participants_Summary.csv")
    master_df.to_csv(master_out, index=False)
    logger.info(f"Saved master summary: {master_out}")

    # Visualizations for selected metrics
    for metric in KEY_METRICS:
        # Bar chart
        out_bar = os.path.join(FIGURES_DIR, f"{metric}_comparison.png")
        plot_bar_paired(master_df, metric, metric.replace("_", " "), out_bar)
        # Paired lines
        out_line = os.path.join(FIGURES_DIR, f"{metric}_paired_lines.png")
        plot_paired_lines(master_df, metric, metric.replace("_", " "), out_line)

    # Plot relative power heatmap
    out_heat = os.path.join(FIGURES_DIR, "EEG_RelativePower_Heatmap.png")
    plot_heatmap_relative_power(master_df, out_heat)

    # Correlation matrix (percent change)
    out_corr = os.path.join(FIGURES_DIR, "Correlation_Matrix.png")
    plot_correlation_matrix(master_df, KEY_METRICS, out_corr)

    # Group statistics
    stats_df = compute_group_statistics(master_df)

    # Write session report
    report_path = os.path.join(METRICS_DIR, "Session_Report.txt")
    write_session_report(stats_df, master_df, report_path)

    logger.info("Batch processing complete.")

# ----------------- CLI Entrypoint -----------------
if __name__ == "__main__":
    logger.info("Starting BrainMat analysis pipeline...")
    batch_process_all(DATA_DIR)
    logger.info("Done.")
