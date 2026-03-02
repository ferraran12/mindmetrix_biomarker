import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def extract_features(physiological: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    print("Extracting features...")
    # Exctract intra subject features
    phase_features = (
        physiological.groupby(["SubjectID", "Phase", "CycleID"])
        .agg(
            PupilMean=("PupilDiameter", "mean"),
            PupilStd=("PupilDiameter", "std"),
            PupilMedian=("PupilDiameter", "median"),
            PupilSkew=("PupilDiameter", "skew"),
            PulseMean=("PulseBPM", "mean"),
            PulseStd=("PulseBPM", "std"),
            PulseMedian=("PulseBPM", "median"),
            PulseSkew=("PulseBPM", "skew"),
            GazeXStd=("GazeX", "std"),
            GazeYStd=("GazeY", "std"),
            GazeZStd=("GazeZ", "std"),
        )
        .reset_index()
    )
    numeric_cols = [
        "PupilMean",
        "PupilStd",
        "PupilMedian",
        "PupilSkew",
        "PulseMean",
        "PulseStd",
        "PulseMedian",
        "PulseSkew",
        "GazeXStd",
        "GazeYStd",
        "GazeZStd",
    ]

    baseline = phase_features[phase_features["Phase"] == "baseline"].set_index(["SubjectID", "CycleID"])[numeric_cols]

    reactivity_frames = []
    for phase in ["relax", "break"]:
        phase_df = phase_features[phase_features["Phase"] == phase].set_index(["SubjectID", "CycleID"])[numeric_cols]
        delta = (phase_df - baseline).add_prefix(f"delta_{phase}_")
        delta_avg = delta.groupby("SubjectID").mean()
        reactivity_frames.append(delta_avg)

    reactivity = pd.concat(reactivity_frames, axis=1)

    cycle_agg = (
        physiological[physiological["Phase"] == "baseline"]
        .groupby(["SubjectID", "CycleID"])
        .agg(
            PupilMean=("PupilDiameter", "mean"),
            PulseMean=("PulseBPM", "mean"),
        )
        .reset_index()
    )

    habituation = cycle_agg.groupby("SubjectID").apply(
        lambda g: pd.Series(
            {
                "PupilSlope": compute_slope(g, "PupilMean", num_points=3),
                "PulseSlope": compute_slope(g, "PulseMean", num_points=3),
            }
        ),
        include_groups=False,
    )

    # Extract inter-subject features
    population_means = phase_features.groupby("Phase")[numeric_cols].mean()

    delta_pop_frames = []
    for phase in ["baseline", "relax", "break"]:
        subj_phase = phase_features[phase_features["Phase"] == phase].set_index(["SubjectID", "CycleID"])[numeric_cols]
        delta = (subj_phase - population_means.loc[phase]).add_prefix(f"deltapop_{phase}_")
        delta_avg = delta.groupby("SubjectID").mean()
        delta_pop_frames.append(delta_avg)

    delta_population = pd.concat(delta_pop_frames, axis=1)

    subject_profile = physiological.groupby("SubjectID").agg(
        PupilMean=("PupilDiameter", "mean"),
        PupilStd=("PupilDiameter", "std"),
        PupilIQR=("PupilDiameter", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        PulseMean=("PulseBPM", "mean"),
        PulseStd=("PulseBPM", "std"),
        PulseIQR=("PulseBPM", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        MotionMean=("MotionMag", "mean"),
        MotionStd=("MotionMag", "std"),
        GazeXSpread=("GazeAngleX", "std"),
        GazeYSpread=("GazeAngleY", "std"),
    )

    # Combine all features
    all_features = subject_profile.join(reactivity).join(delta_population).join(habituation)
    all_features = all_features.join(
        subjects.set_index("SubjectID")[
            ["STAI_T", "STAI_S", "Gender", "Handedness", "WearsGlasses", "CalibrationError", "BloodType"]
        ]
    )
    print("Features extracted successfully.")
    all_features.to_csv("data/extracted/features.csv", index=False)
    print("Features saved to data/extracted/features.csv")
    return all_features


def compute_slope(group: pd.DataFrame, col: str, num_points: int = 3) -> float:
    if len(group) < num_points:
        return np.nan
    slope, _, _, _, _ = stats.linregress(group["CycleID"], group[col])
    return slope


if __name__ == "__main__":
    physiological = pd.read_csv("data/preprocessed/physiological.csv")
    subjects = pd.read_csv("data/preprocessed/subjects.csv")
    all_features = extract_features(physiological, subjects)
