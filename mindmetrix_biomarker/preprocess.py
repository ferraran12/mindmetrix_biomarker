import pandas as pd
import pandera.pandas as pa
import numpy as np
from pathlib import Path

from mindmetrix_biomarker.loader import load_data


def preprocess_physiological_data(
    df: pd.DataFrame,
    *,
    PPG_SQI_threshold: float = 0.5,
    MotionMag_threshold: float = 0.7,
    PupilDiameter_lower_threshold: float = 2.0,
    PupilDiameter_upper_threshold: float = 9.0,
    PulseBPM_lower_threshold: float = 40.0,
    PulseBPM_upper_threshold: float = 300.0,
) -> pd.DataFrame:
    # Check if df.empty:
    if df.empty:
        print("Warning: The input DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if the input is empty

    # Check data schema
    # SubjectID  DeviceTimestamp  CycleID     Phase  PupilDiameter     GazeX
    # GazeY     GazeZ   PulseBPM   PPG_SQI  MotionMag
    schema: pa.DataFrameSchema = pa.DataFrameSchema(
        {
            "SubjectID": pa.Column(str),
            "DeviceTimestamp": pa.Column(int),
            "CycleID": pa.Column(int, pa.Check.isin(range(1, 11))),
            "Phase": pa.Column(str, pa.Check.isin(["baseline", "relax", "break"])),
            "PupilDiameter": pa.Column(float),
            "GazeX": pa.Column(float),
            "GazeY": pa.Column(float),
            "GazeZ": pa.Column(float),
            "PulseBPM": pa.Column(float),
            "PPG_SQI": pa.Column(float, pa.Check(lambda s: 0 <= s.all() <= 1, ignore_na=True)),
            "MotionMag": pa.Column(float),
        }
    )
    try:
        schema.validate(df)
    except pa.errors.SchemaError as e:
        print(f"Schema validation error: {e}")
        raise e

    # Remove rows with missing values
    df: pd.DataFrame = df.dropna()
    # Delete duplicates
    df: pd.DataFrame = df.drop_duplicates()
    # Remove implausible values in physiological data
    df: pd.DataFrame = df[
        (df.PupilDiameter >= PupilDiameter_lower_threshold) & (df.PupilDiameter <= PupilDiameter_upper_threshold)
    ]
    df: pd.DataFrame = df[(df.PulseBPM >= PulseBPM_lower_threshold) & (df.PulseBPM <= PulseBPM_upper_threshold)]
    # Discart low quality pulse signals
    df: pd.DataFrame = df[df.PPG_SQI > PPG_SQI_threshold]
    # Discart too high motion magnitude
    df: pd.DataFrame = df[df.MotionMag <= MotionMag_threshold]
    # Filter by angles
    df["GazeAngleX"] = np.degrees(np.arctan2(df["GazeX"], df["GazeZ"]))
    df["GazeAngleY"] = np.degrees(np.arctan2(df["GazeY"], df["GazeZ"]))
    df: pd.DataFrame = df[(df["GazeAngleX"].between(-30, 30)) & (df["GazeAngleY"].between(-30, 30))]
    #Enforce correct data types
    df["SubjectID"] = df["SubjectID"].astype(str)
    df["DeviceTimestamp"] = df["DeviceTimestamp"].astype(int)
    df["CycleID"] = df["CycleID"].astype(int)
    df["Phase"] = df["Phase"].astype(str)
    df["PupilDiameter"] = df["PupilDiameter"].astype(float)
    df["GazeX"] = df["GazeX"].astype(float)
    df["GazeY"] = df["GazeY"].astype(float)
    df["GazeZ"] = df["GazeZ"].astype(float)
    df["PulseBPM"] = df["PulseBPM"].astype(float)
    df["PPG_SQI"] = df["PPG_SQI"].astype(float)
    df["MotionMag"] = df["MotionMag"].astype(float)

    return df


def preprocess_subjects_data(df: pd.DataFrame) -> pd.DataFrame:
    # Check if df.empty:
    if df.empty:
        print("Warning: The input DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if the input is empty
    # Check data schema
    # SubjectID,STAI_T,STAI_S,Gender,
    # Handedness,WearsGlasses,CalibrationError,BloodType
    schema: pa.DataFrameSchema = pa.DataFrameSchema(
        {
            "SubjectID": pa.Column(str),
            "STAI_T": pa.Column(int),
            "STAI_S": pa.Column(int),
            "Gender": pa.Column(str, pa.Check.isin(["F", "M"], ignore_na=True)),
            "Handedness": pa.Column(str, pa.Check.isin(["R", "L"], ignore_na=True)),
            "WearsGlasses": pa.Column(int, pa.Check.isin([0, 1], ignore_na=True)),
            "CalibrationError": pa.Column(float),
            "BloodType": pa.Column(str, pa.Check.isin(["A", "B", "AB", "O"], ignore_na=True)),
        }
    )
    try:
        schema.validate(df)
    except pa.errors.SchemaError as e:
        print(f"Schema validation error: {e}")
    # Remove rows with missing values
    df: pd.DataFrame = df.dropna()
    # Delete duplicates
    df: pd.DataFrame = df.drop_duplicates()
    # Enforce correct data types
    df["STAI_T"] = df["STAI_T"].astype(int)
    df["STAI_S"] = df["STAI_S"].astype(int)
    df["Gender"] = df["Gender"].astype(str)
    df["Handedness"] = df["Handedness"].astype(str)
    df["WearsGlasses"] = df["WearsGlasses"].astype(int)
    df["CalibrationError"] = df["CalibrationError"].astype(float)
    df["BloodType"] = df["BloodType"].astype(str)

    return df


if __name__ == "__main__":
    data_path = Path("data/random_sample_timeseries.csv")
    df = load_data(data_path)
    preprocessed_physiological_df = preprocess_physiological_data(df)
    # preprocessed_physiological_df.to_csv("data/preprocessed_physiological_data.csv", index=False)
    subjects_data_path = Path("data/subjects.csv")
    subjects_df = load_data(subjects_data_path)
    preprocessed_subjects_df = preprocess_subjects_data(subjects_df)
    preprocessed_subjects_df.to_csv("data/preprocessed_subjects_data.csv", index=False)
