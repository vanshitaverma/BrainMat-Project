import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
DATA_PATH = "./data/Questionnaire.csv"
OUT_DIR = "./results/questionnaire"
METRICS = f"{OUT_DIR}/metrics"
FIGS = f"{OUT_DIR}/figures"

os.makedirs(METRICS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

sns.set(style="whitegrid")

# ---------------- LOAD ----------------
# df = pd.read_csv(DATA_PATH)
# df = pd.read_csv(DATA_PATH, encoding="latin1")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1",
    engine="python",
    sep=",",
    quotechar='"',
    on_bad_lines="skip"
)


# Rename columns for sanity
# df = df.rename(columns={
#     "I felt more relaxed after the session.": "Relaxed",
#     "The walking felt calming or meditative.": "Calming",
#     "I felt more focused and clear-minded.": "Focused",
#     "The mat felt comfortable under my feet.": "Comfort",
#     "The overall experience was enjoyable.": "Enjoyment",
#     "How would you rate your mood before the session?": "Mood_Before",
#     "How would you rate your mood after the session?": "Mood_After",
#     "Did you notice any change in your energy or alertness levels?": "Energy",
#     "Did you feel any meditative or mindful state during the session?": "Meditative"
# })

# # -------- AUTO COLUMN MAPPING --------

# def find_col(keyword):
#     for c in df.columns:
#         if keyword.lower() in c.lower():
#             return c
#     return None

# COL_MAP = {
#     "Relaxed": find_col("relaxed"),
#     "Calming": find_col("calming"),
#     "Focused": find_col("focused"),
#     "Comfort": find_col("comfortable"),
#     "Enjoyment": find_col("enjoyable"),
#     "Mood_Before": find_col("mood before"),
#     "Mood_After": find_col("mood after"),
#     "Energy": find_col("energy"),
#     "Meditative": find_col("meditative")
# }

# for k,v in COL_MAP.items():
#     if v is not None:
#         df = df.rename(columns={v:k})
#     else:
#         print(f"⚠️ Could not find column for {k}")


# LIKERT = ["Relaxed","Calming","Focused","Comfort","Enjoyment"]

# ---------------- CLEAN + STANDARDIZE ----------------

RELAXED_COL = "I felt more relaxed after the session."
CALM_COL = "The walking felt calming or meditative."
FOCUS_COL = "I felt more focused and clear-minded."
COMFORT_COL = "The mat felt comfortable under my feet."
ENJOY_COL = "The overall experience was enjoyable."
MOOD_BEFORE_COL = "How would you rate your mood before the session?"
MOOD_AFTER_COL = "How would you rate your mood after the session?"
ENERGY_COL = "Did you notice any change in your energy or alertness levels?"
MEDITATIVE_COL = "Did you feel any meditative or mindful state during the session?"

# Create simplified numeric columns (do NOT overwrite originals)
df["Relaxed"] = pd.to_numeric(df[RELAXED_COL], errors="coerce")
df["Calming"] = pd.to_numeric(df[CALM_COL], errors="coerce")
df["Focused"] = pd.to_numeric(df[FOCUS_COL], errors="coerce")
df["Comfort"] = pd.to_numeric(df[COMFORT_COL], errors="coerce")
df["Enjoyment"] = pd.to_numeric(df[ENJOY_COL], errors="coerce")

df["Mood_Before"] = pd.to_numeric(df[MOOD_BEFORE_COL], errors="coerce")
df["Mood_After"] = pd.to_numeric(df[MOOD_AFTER_COL], errors="coerce")

LIKERT = ["Relaxed","Calming","Focused","Comfort","Enjoyment"]

# Energy mapping
ENERGY_MAP = {
    "Decreased": -1,
    "No change": 0,
    "Slightly increased": 1,
    "Significantly increased": 2
}

df["Energy_Num"] = df[ENERGY_COL].map(ENERGY_MAP)

# Meditative mapping
MED_MAP = {
    "Not at all": 1,
    "Slightly": 2,
    "Moderately": 3,
    "Deeply": 4
}

df["Meditative_Num"] = df[MEDITATIVE_COL].map(MED_MAP)

# Mood change
df["Mood_Change"] = df["Mood_After"] - df["Mood_Before"]


# # Convert Likert columns to numeric
# for c in LIKERT + ["Mood_Before","Mood_After"]:
#     df[c] = pd.to_numeric(df[c], errors="coerce")

# # Energy mapping
# ENERGY_MAP = {
#     "Decreased": -1,
#     "No change": 0,
#     "Slightly increased": 1,
#     "Significantly increased": 2
# }

# df["Energy_Num"] = df["Energy"].map(ENERGY_MAP)

# # Meditative mapping
# MED_MAP = {
#     "Not at all": 1,
#     "Slightly": 2,
#     "Moderately": 3,
#     "Deeply": 4
# }
# df["Meditative_Num"] = df["Meditative"].map(MED_MAP)

# # Mood change
# df["Mood_Change"] = df["Mood_After"] - df["Mood_Before"]

# ---------------- SUMMARY TABLES ----------------

likert_summary = df[LIKERT].mean().round(2)
likert_summary.to_csv(f"{METRICS}/Likert_Summary.csv")

mood_stats = pd.DataFrame({
    "Mood_Before_Mean":[df["Mood_Before"].mean()],
    "Mood_After_Mean":[df["Mood_After"].mean()],
    "Mood_Change_Mean":[df["Mood_Change"].mean()]
}).round(2)

mood_stats.to_csv(f"{METRICS}/Mood_Stats.csv", index=False)

energy_dist = df["Energy_Num"].value_counts().sort_index()
energy_dist.to_csv(f"{METRICS}/Energy_Distribution.csv")

# ---------------- PLOTS ----------------

# Likert Bars
plt.figure(figsize=(8,5))
likert_summary.plot(kind="bar")
plt.ylabel("Mean Score (1–5)")
plt.title("Mean Subjective Ratings")
plt.tight_layout()
plt.savefig(f"{FIGS}/Likert_Bars.png")
plt.close()

# Mood Before vs After
plt.figure(figsize=(5,4))
plt.bar(["Before","After"],
        [df["Mood_Before"].mean(), df["Mood_After"].mean()])
plt.ylabel("Mood Rating")
plt.title("Mood Before vs After Session")
plt.tight_layout()
plt.savefig(f"{FIGS}/Mood_Before_After.png")
plt.close()

# Energy Change
plt.figure(figsize=(5,4))
energy_dist.plot(kind="bar")
plt.xticks(ticks=[0,1,2,3], labels=["↓","0","+","++"], rotation=0)
plt.title("Energy / Alertness Change")
plt.tight_layout()
plt.savefig(f"{FIGS}/Energy_Change.png")
plt.close()

# Meditative State
plt.figure(figsize=(5,4))
df["Meditative_Num"].value_counts().sort_index().plot(kind="bar")
plt.xticks(ticks=[0,1,2,3], labels=["Not","Slight","Moderate","Deep"], rotation=0)
plt.title("Reported Meditative State")
plt.tight_layout()
plt.savefig(f"{FIGS}/Meditative_State.png")
plt.close()

# Comfort vs Enjoyment
plt.figure(figsize=(5,4))
plt.scatter(df["Comfort"], df["Enjoyment"])
plt.xlabel("Comfort")
plt.ylabel("Enjoyment")
plt.title("Comfort vs Enjoyment")
plt.tight_layout()
plt.savefig(f"{FIGS}/Comfort_Enjoyment.png")
plt.close()

print("✅ Questionnaire analysis complete.")
print(f"📂 Outputs saved to {OUT_DIR}")
