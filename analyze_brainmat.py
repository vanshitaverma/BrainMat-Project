import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./data"
RESULTS_DIR = "./results"

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

# ------------------------------------------------------------
# OPTIONAL SIGNAL QUALITY FILTERING
# ------------------------------------------------------------
# Set to True if you want to remove samples with poor/no HSI.
#
# HSI values:
#   1 = good connection
#   2 = fair connection
#
# Therefore, when filtering is enabled, values 1 AND 2 are kept.
#
# Set to False to analyse all available EEG samples.
# ------------------------------------------------------------

USE_HSI_FILTER = False

HSI_ACCEPTED_VALUES = [1, 2]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(file_path):
    """
    Load Muse CSV file and clean missing/invalid values.

    Event-only rows from the Muse recording contain no EEG
    values and are retained initially, but rows containing
    no numerical data will not contribute to numerical means.
    """

    df = pd.read_csv(file_path)

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Convert EEG / physiological columns to numeric
    numerical_columns = []

    for band in BANDS:
        for channel in CHANNELS:
            column = f"{band}_{channel}"

            if column in df.columns:
                numerical_columns.append(column)

    additional_columns = [
        "Heart_Rate",
        "Accelerometer_X",
        "Accelerometer_Y",
        "Accelerometer_Z",
        "Gyro_X",
        "Gyro_Y",
        "Gyro_Z"
    ]

    for column in additional_columns:
        if column in df.columns:
            numerical_columns.append(column)

    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# ============================================================
# HSI FILTERING
# ============================================================

def apply_hsi_filter(df):
    """
    Optionally filter EEG samples using Muse HSI values.

    HSI values 1 and 2 are treated as acceptable.

    A row is retained only if all four EEG electrode HSI
    values are within the accepted range.
    """

    hsi_columns = [
        f"HSI_{channel}"
        for channel in CHANNELS
        if f"HSI_{channel}" in df.columns
    ]

    # If HSI filtering is disabled, return original data
    if not USE_HSI_FILTER:
        return df

    # If HSI columns are unavailable, do not filter
    if len(hsi_columns) != len(CHANNELS):
        print("⚠️ HSI columns incomplete. HSI filtering skipped.")
        return df

    # Convert HSI values to numeric
    for column in hsi_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep rows where ALL four electrodes have HSI 1 or 2
    valid_rows = df[hsi_columns].isin(HSI_ACCEPTED_VALUES).all(axis=1)

    filtered_df = df.loc[valid_rows].copy()

    return filtered_df


# ============================================================
# BAND POWER
# ============================================================

def compute_band_means(df):
    """
    Compute mean absolute EEG band power across the four
    Muse channels.

    For each band:
        1. Calculate the mean across channels for each sample.
        2. Calculate the mean across all valid samples.
    """

    results = {}

    for band in BANDS:

        columns = [
            f"{band}_{channel}"
            for channel in CHANNELS
            if f"{band}_{channel}" in df.columns
        ]

        if not columns:
            results[f"{band}_mean"] = np.nan
            continue

        # Mean across channels for every sample
        sample_means = df[columns].mean(axis=1, skipna=True)

        # Mean across samples
        results[f"{band}_mean"] = sample_means.mean()

    return results


# ============================================================
# RELATIVE BAND POWER
# ============================================================

def compute_relative_band_power(band_means):
    """
    Calculate relative EEG band power.

    Relative power =
        band power / total power

    where total power is the sum of Delta, Theta, Alpha,
    Beta and Gamma.
    """

    relative = {}

    available_values = [
        band_means.get(f"{band}_mean", np.nan)
        for band in BANDS
    ]

    if all(pd.isna(value) for value in available_values):
        return {
            f"{band}_relative": np.nan
            for band in BANDS
        }

    total_power = np.nansum(available_values)

    if total_power == 0 or np.isnan(total_power):
        return {
            f"{band}_relative": np.nan
            for band in BANDS
        }

    for band in BANDS:

        value = band_means.get(f"{band}_mean", np.nan)

        if pd.isna(value):
            relative[f"{band}_relative"] = np.nan
        else:
            relative[f"{band}_relative"] = value / total_power

    return relative


# ============================================================
# EEG RATIOS / INDICES
# ============================================================

def compute_ratios(band_means):
    """
    Compute derived EEG indices.

    Indices:

    Theta/Beta Ratio
        Theta / Beta

    Alpha/Delta Ratio
        Alpha / Delta

    Meditation Index
        (Theta + Alpha) / Beta

    Engagement Index
        Beta / (Alpha + Theta)
    """

    ratios = {}

    theta = band_means.get("Theta_mean", np.nan)
    alpha = band_means.get("Alpha_mean", np.nan)
    beta = band_means.get("Beta_mean", np.nan)
    delta = band_means.get("Delta_mean", np.nan)

    # --------------------------------------------------------
    # Theta / Beta Ratio
    # --------------------------------------------------------

    if (
        not pd.isna(theta)
        and not pd.isna(beta)
        and beta != 0
    ):
        ratios["Theta_Beta_Ratio"] = theta / beta
    else:
        ratios["Theta_Beta_Ratio"] = np.nan

    # --------------------------------------------------------
    # Alpha / Delta Ratio
    # --------------------------------------------------------

    if (
        not pd.isna(alpha)
        and not pd.isna(delta)
        and delta != 0
    ):
        ratios["Alpha_Delta_Ratio"] = alpha / delta
    else:
        ratios["Alpha_Delta_Ratio"] = np.nan

    # --------------------------------------------------------
    # Meditation Index
    # --------------------------------------------------------

    if (
        not pd.isna(theta)
        and not pd.isna(alpha)
        and not pd.isna(beta)
        and beta != 0
    ):
        ratios["Meditation_Index"] = (theta + alpha) / beta
    else:
        ratios["Meditation_Index"] = np.nan

    # --------------------------------------------------------
    # Engagement Index
    # --------------------------------------------------------

    denominator = np.nan

    if not pd.isna(alpha) and not pd.isna(theta):
        denominator = alpha + theta

    if (
        not pd.isna(beta)
        and not pd.isna(denominator)
        and denominator != 0
    ):
        ratios["Engagement_Index"] = beta / denominator
    else:
        ratios["Engagement_Index"] = np.nan

    return ratios


# ============================================================
# FRONTAL ALPHA ASYMMETRY
# ============================================================

def compute_frontal_alpha_asymmetry(df):
    """
    Calculate Frontal Alpha Asymmetry (FAA).

    FAA = log(Alpha_AF8) - log(Alpha_AF7)

    This is calculated sample-wise and then averaged
    across the session.

    Zero or negative values are excluded because the
    logarithm is undefined for those values.
    """

    required_columns = [
        "Alpha_AF7",
        "Alpha_AF8"
    ]

    if not all(column in df.columns for column in required_columns):
        return np.nan

    af7 = pd.to_numeric(
        df["Alpha_AF7"],
        errors="coerce"
    )

    af8 = pd.to_numeric(
        df["Alpha_AF8"],
        errors="coerce"
    )

    # Only positive values can be log-transformed
    af7 = af7.where(af7 > 0)
    af8 = af8.where(af8 > 0)

    faa = np.log(af8) - np.log(af7)

    return faa.mean()


# ============================================================
# ACCELEROMETER MAGNITUDE
# ============================================================

def compute_accelerometer_magnitude(df):
    """
    Calculate mean accelerometer magnitude.

    Magnitude:

        sqrt(X^2 + Y^2 + Z^2)

    The result represents the overall magnitude of
    head movement / acceleration.
    """

    required_columns = [
        "Accelerometer_X",
        "Accelerometer_Y",
        "Accelerometer_Z"
    ]

    if not all(column in df.columns for column in required_columns):
        return np.nan

    x = pd.to_numeric(
        df["Accelerometer_X"],
        errors="coerce"
    )

    y = pd.to_numeric(
        df["Accelerometer_Y"],
        errors="coerce"
    )

    z = pd.to_numeric(
        df["Accelerometer_Z"],
        errors="coerce"
    )

    magnitude = np.sqrt(
        x**2 + y**2 + z**2
    )

    return magnitude.mean()


# ============================================================
# HEART RATE
# ============================================================

def compute_heart_rate(df):
    """
    Calculate mean heart rate for the session.
    """

    if "Heart_Rate" not in df.columns:
        return np.nan

    heart_rate = pd.to_numeric(
        df["Heart_Rate"],
        errors="coerce"
    )

    return heart_rate.mean()


# ============================================================
# SESSION SUMMARY
# ============================================================

def summarize_file(file_path):
    """
    Generate a complete summary for one Muse recording.

    The same function is used for both BEFORE and DURING
    walking recordings.
    """

    print(f"   Processing: {file_path}")

    # Load data
    df = load_csv(file_path)

    # --------------------------------------------------------
    # HSI filtering
    # --------------------------------------------------------

    original_rows = len(df)

    df = apply_hsi_filter(df)

    filtered_rows = len(df)

    # --------------------------------------------------------
    # EEG band powers
    # --------------------------------------------------------

    band_means = compute_band_means(df)

    # --------------------------------------------------------
    # Relative EEG powers
    # --------------------------------------------------------

    relative_power = compute_relative_band_power(
        band_means
    )

    # --------------------------------------------------------
    # EEG indices
    # --------------------------------------------------------

    ratios = compute_ratios(
        band_means
    )

    # --------------------------------------------------------
    # Frontal Alpha Asymmetry
    # --------------------------------------------------------

    frontal_alpha_asymmetry = (
        compute_frontal_alpha_asymmetry(df)
    )

    # --------------------------------------------------------
    # Heart rate
    # --------------------------------------------------------

    heart_rate = compute_heart_rate(df)

    # --------------------------------------------------------
    # Accelerometer
    # --------------------------------------------------------

    accelerometer_magnitude = (
        compute_accelerometer_magnitude(df)
    )

    # --------------------------------------------------------
    # Combine all results
    # --------------------------------------------------------

    summary = {
        **band_means,
        **relative_power,
        **ratios,

        "Frontal_Alpha_Asymmetry":
            frontal_alpha_asymmetry,

        "HeartRate_mean":
            heart_rate,

        "Accelerometer_Magnitude_mean":
            accelerometer_magnitude,

        "Rows_original":
            original_rows,

        "Rows_after_HSI_filter":
            filtered_rows
    }

    return summary


# ============================================================
# BEFORE vs DURING COMPARISON
# ============================================================

def compare_sessions(
    before_csv,
    during_csv,
    participant_id
):
    """
    Compare baseline (before walking) and walking
    sessions for one participant.

    For every metric:

        change = during - before
    """

    print(
        f"\nParticipant {participant_id}"
    )

    # --------------------------------------------------------
    # Analyse sessions
    # --------------------------------------------------------

    before_summary = summarize_file(
        before_csv
    )

    during_summary = summarize_file(
        during_csv
    )

    # --------------------------------------------------------
    # Calculate changes
    # --------------------------------------------------------

    comparison = {}

    for key in before_summary:

        before_value = before_summary.get(
            key,
            np.nan
        )

        during_value = during_summary.get(
            key,
            np.nan
        )

        comparison[
            f"{key}_before"
        ] = before_value

        comparison[
            f"{key}_during"
        ] = during_value

        # Numeric change
        if (
            not pd.isna(before_value)
            and not pd.isna(during_value)
        ):

            comparison[
                f"{key}_change"
            ] = during_value - before_value

        else:

            comparison[
                f"{key}_change"
            ] = np.nan

    comparison[
        "Participant_ID"
    ] = participant_id

    return comparison


# ============================================================
# INDIVIDUAL PARTICIPANT OUTPUT
# ============================================================

def run_analysis(
    before_csv,
    during_csv,
    participant_id
):
    """
    Analyse one participant and save their summary.
    """

    summary = compare_sessions(
        before_csv,
        during_csv,
        participant_id
    )

    summary_df = pd.DataFrame(
        [summary]
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        RESULTS_DIR,
        f"{participant_id}_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"✅ Processed {participant_id}"
    )

    print(
        f"   Saved: {output_path}"
    )

    return summary_df


# ============================================================
# GROUP-LEVEL STATISTICS
# ============================================================

def calculate_group_statistics(df, metrics):
    """
    Calculate group-level before vs during statistics.

    Statistics calculated:
    - Before mean
    - During mean
    - Mean absolute change
    - Mean percentage change
    - Paired t-test
    - Cohen's d
    - 95% confidence interval
    """

    results = []

    for metric in metrics:

        before_col = f"{metric}_before"
        during_col = f"{metric}_during"

        # Check that required columns exist
        if before_col not in df.columns or during_col not in df.columns:
            print(f"⚠️ Skipping {metric}: columns not found.")
            continue

        # Keep only participants with both measurements
        valid = df[[before_col, during_col]].dropna()

        if len(valid) < 2:
            print(f"⚠️ Skipping {metric}: insufficient paired data.")
            continue

        before = valid[before_col].astype(float)
        during = valid[during_col].astype(float)

        n = len(valid)

        # ------------------------------------------------
        # Participant-level change
        # ------------------------------------------------

        change = during - before

        # ------------------------------------------------
        # Percentage change
        # ------------------------------------------------

        percent_change = np.where(
            before != 0,
            ((during - before) / before) * 100,
            np.nan
        )

        percent_change = pd.Series(percent_change).dropna()

        # ------------------------------------------------
        # Descriptive statistics
        # ------------------------------------------------

        before_mean = before.mean()
        during_mean = during.mean()
        mean_change = change.mean()
        sd_change = change.std(ddof=1)

        # ------------------------------------------------
        # Paired t-test
        # ------------------------------------------------

        t_stat, p_value = stats.ttest_rel(
            during,
            before
        )

        # ------------------------------------------------
        # Cohen's d for paired data
        # ------------------------------------------------

        if sd_change != 0:
            cohens_d = mean_change / sd_change
        else:
            cohens_d = np.nan

        # ------------------------------------------------
        # 95% confidence interval
        # ------------------------------------------------

        standard_error = sd_change / np.sqrt(n)

        t_critical = stats.t.ppf(
            0.975,
            df=n - 1
        )

        ci_lower = mean_change - (
            t_critical * standard_error
        )

        ci_upper = mean_change + (
            t_critical * standard_error
        )

        # ------------------------------------------------
        # Mean percentage change
        # ------------------------------------------------

        mean_percent_change = percent_change.mean()

        # ------------------------------------------------
        # Store results
        # ------------------------------------------------

        results.append({
            "Metric": metric,
            "N": n,
            "Before_Mean": before_mean,
            "During_Mean": during_mean,
            "Mean_Change": mean_change,
            "Mean_Percent_Change": mean_percent_change,
            "t_statistic": t_stat,
            "p_value": p_value,
            "Cohens_d": cohens_d,
            "CI_95_Lower": ci_lower,
            "CI_95_Upper": ci_upper
        })

    return pd.DataFrame(results)

# ============================================================
# METRICS FOR GROUP ANALYSIS
# ============================================================

GROUP_METRICS = [
    "Delta_mean",
    "Theta_mean",
    "Alpha_mean",
    "Beta_mean",
    "Gamma_mean",

    "Theta_Beta_Ratio",
    "Alpha_Delta_Ratio",
    "Meditation_Index",
    "Engagement_Index",
    "Frontal_Alpha_Asymmetry",

    "HeartRate_mean",
    "Accelerometer_Magnitude"
]


# ============================================================
# FDR CORRECTION
# ============================================================

def apply_fdr_correction(stats_df):
    """
    Apply Benjamini-Hochberg False Discovery Rate correction
    to the group-level p-values.
    """

    if stats_df.empty:
        return stats_df

    reject, corrected_p, _, _ = multipletests(
        stats_df["p_value"],
        alpha=0.05,
        method="fdr_bh"
    )

    stats_df["p_value_FDR"] = corrected_p
    stats_df["Significant_FDR"] = reject

    return stats_df


# ============================================================
# FIGURE GENERATION
# ============================================================
def generate_group_statistics_figures(
    group_stats,
    output_dir
):
    """
    Generate group-level figures showing mean change
    from before to during walking.

    Error bars represent 95% confidence intervals.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for _, row in group_stats.iterrows():

        metric = row["Metric"]

        mean_change = row["Mean_Change"]

        ci_lower = row["CI_95_Lower"]
        ci_upper = row["CI_95_Upper"]

        # Skip if statistics are unavailable
        if pd.isna(mean_change):
            continue

        # ------------------------------------------------
        # Calculate asymmetric error bars
        # ------------------------------------------------

        lower_error = (
            mean_change - ci_lower
        )

        upper_error = (
            ci_upper - mean_change
        )

        # ------------------------------------------------
        # Create figure
        # ------------------------------------------------

        plt.figure(
            figsize=(7, 5)
        )

        plt.errorbar(
            1,
            mean_change,
            yerr=[
                [lower_error],
                [upper_error]
            ],
            fmt="o",
            capsize=6,
            markersize=8
        )

        # Zero = no change
        plt.axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        plt.xticks(
            [1],
            ["During vs Before"]
        )

        plt.ylabel(
            "Mean Change"
        )

        plt.title(
            metric.replace("_", " ")
        )

        # ------------------------------------------------
        # Add statistical information
        # ------------------------------------------------

        p_value = row["p_value"]
        cohens_d = row["Cohens_d"]

        plt.text(
            1,
            ci_upper,
            (
                f"p = {p_value:.4f}\n"
                f"Cohen's d = {cohens_d:.3f}"
            ),
            ha="center",
            va="bottom"
        )

        plt.tight_layout()

        # ------------------------------------------------
        # Save figure
        # ------------------------------------------------

        filename = (
            metric +
            "_group_statistics.png"
        )

        filepath = os.path.join(
            output_dir,
            filename
        )

        plt.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"📈 Saved figure: {filepath}"
        )


# ============================================================
# BATCH PROCESSING
# ============================================================

def batch_process():
    """
    Automatically detect and process all participants.

    Expected naming convention:

        P01_before.csv
        P01_during.csv

        P02_before.csv
        P02_during.csv

        etc.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find all CSV files
    # --------------------------------------------------------

    files = glob.glob(
        os.path.join(
            DATA_DIR,
            "*.csv"
        )
    )

    if not files:

        print(
            "❌ No CSV files found in:"
        )

        print(
            f"   {DATA_DIR}"
        )

        return

    # --------------------------------------------------------
    # Extract participant IDs
    # --------------------------------------------------------

    participants = set()

    for file_path in files:

        filename = os.path.basename(
            file_path
        )

        # Example:
        # P01_before.csv

        if "_before.csv" in filename:

            participant_id = (
                filename
                .replace("_before.csv", "")
            )

            participants.add(
                participant_id
            )

        elif "_during.csv" in filename:

            participant_id = (
                filename
                .replace("_during.csv", "")
            )

            participants.add(
                participant_id
            )

    participants = sorted(
        participants
    )

    print(
        "\n========================================"
    )

    print(
        "BrainMat Batch Analysis"
    )

    print(
        "========================================"
    )

    print(
        f"Data directory: {DATA_DIR}"
    )

    print(
        f"Results directory: {RESULTS_DIR}"
    )

    print(
        f"HSI filtering: {USE_HSI_FILTER}"
    )

    if USE_HSI_FILTER:

        print(
            f"Accepted HSI values: "
            f"{HSI_ACCEPTED_VALUES}"
        )

    print(
        f"\nFound {len(participants)} "
        f"participants:"
    )

    print(
        participants
    )

    # --------------------------------------------------------
    # Process participants
    # --------------------------------------------------------

    all_summaries = []

    missing_participants = []

    for participant_id in participants:

        before_csv = os.path.join(
            DATA_DIR,
            f"{participant_id}_before.csv"
        )

        during_csv = os.path.join(
            DATA_DIR,
            f"{participant_id}_during.csv"
        )

        # ----------------------------------------------------
        # Check both files exist
        # ----------------------------------------------------

        if (
            os.path.exists(before_csv)
            and os.path.exists(during_csv)
        ):

            summary_df = run_analysis(
                before_csv,
                during_csv,
                participant_id
            )

            all_summaries.append(
                summary_df
            )

        else:

            print(
                f"\n⚠️ Missing data for "
                f"{participant_id}"
            )

            if not os.path.exists(
                before_csv
            ):

                print(
                    "   Missing BEFORE file"
                )

            if not os.path.exists(
                during_csv
            ):

                print(
                    "   Missing DURING file"
                )

            missing_participants.append(
                participant_id
            )

    # ========================================================
    # COMBINE ALL PARTICIPANTS
    # ========================================================

    if all_summaries:

        combined_df = pd.concat(
            all_summaries,
            ignore_index=True
        )

        # ====================================================
        # GROUP-LEVEL STATISTICS
        # ====================================================

        group_metrics = [
            "Alpha_mean",
            "Theta_mean",
            "Beta_mean",
            "Theta_Beta_Ratio",
            "Alpha_Delta_Ratio",
            "Meditation_Index",
            "Engagement_Index",
            "Frontal_Alpha_Asymmetry",
            "HeartRate_mean",
            "Accelerometer_Magnitude_mean"
        ]

        group_stats = calculate_group_statistics(
            combined_df,
            group_metrics
        )

        # Save group-level statistics
        group_stats_path = os.path.join(
            RESULTS_DIR,
            "Group_Statistics.csv"
        )

        group_stats.to_csv(
            group_stats_path,
            index=False
        )

        print(
            f"\n📊 Group statistics saved to:"
        )

        print(
            f"   {group_stats_path}"
        )

        # calculate group stats
        # calculate_group_statistics(
        #     combined_df
        # )


        # ========================================================
        # GROUP-LEVEL FIGURES
        # ========================================================

        group_figures_dir = os.path.join(
            RESULTS_DIR,
            "group_statistics_figures"
        )

        generate_group_statistics_figures(
            group_stats,
            group_figures_dir
        )

        # --------------------------------------------------------
        # Put Participant_ID first
        # --------------------------------------------------------


        # Put Participant_ID first
        columns = (
            ["Participant_ID"]
            +
            [
                column
                for column in combined_df.columns
                if column != "Participant_ID"
            ]
        )

        combined_df = combined_df[
            columns
        ]

        master_path = os.path.join(
            RESULTS_DIR,
            "All_Participants_Summary.csv"
        )

        combined_df.to_csv(
            master_path,
            index=False
        )

        print(
            "\n========================================"
        )

        print(
            "MASTER RESULTS"
        )

        print(
            "========================================"
        )

        print(
            f"Participants successfully "
            f"processed: "
            f"{len(all_summaries)}"
        )

        print(
            f"Master summary saved to:"
        )

        print(
            f"   {master_path}"
        )

    else:

        print(
            "\n❌ No complete participant "
            "datasets found."
        )

    # ========================================================
    # MISSING PARTICIPANT REPORT
    # ========================================================

    if missing_participants:

        missing_df = pd.DataFrame({
            "Participant_ID":
                missing_participants
        })

        missing_path = os.path.join(
            RESULTS_DIR,
            "Missing_Participants.csv"
        )

        missing_df.to_csv(
            missing_path,
            index=False
        )

        print(
            f"\n⚠️ Missing participant report:"
        )

        print(
            f"   {missing_path}"
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "Analysis complete."
    )

    print(
        "========================================\n"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    batch_process()