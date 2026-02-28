import mindmetrix_biomarker.preprocess as preprocess
import pandas as pd

def test_preprocess_physiological_data() -> None:
    df = pd.read_csv("tests/data/timeseries.csv")
    preprocessed_df = preprocess.preprocess_physiological_data(df)
    expected_df = pd.read_csv("tests/data/timeseries_clean.csv")
    # Verify that the preprocessed DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(preprocessed_df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_preprocess_subjects_data() -> None:
    df = pd.read_csv("tests/data/subjects.csv")
    preprocessed_df = preprocess.preprocess_subjects_data(df)
    expected_df = pd.read_csv("tests/data/subjects_clean.csv")
    # Verify that the preprocessed DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(preprocessed_df.reset_index(drop=True), expected_df.reset_index(drop=True))
