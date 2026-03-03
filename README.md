## Getting Started

### Install the dependencies and the mindmetrix_biomarker package
To use the mindmetrix_biomarker package, follow the instructions below to set up your environment and install the package:

```bash
uv sync
source .venv/bin/activate
uv pip install -e .
```
### ⚠️ Insert timeseries.csv data into data/raw folder
Given the size of the data, only the subjects.csv data are already in the `data/raw` folder. Please also insert the `timeseries.csv` into the folder `data/raw`

## Run the analysis
The following notebooks include the data exploration and the results of the analysis.

### Exploration
In `exploration/exploration.ipynb` you find the notebook that explores the datasets.

### Results
In the `results` directory, you can find three Jupyter notebooks: 
- `biomarkers.ipynb` shows the results and answers the hypothesis stated in the Report Mindmetrix Assignment.docx file.
- `PCA.ipynb` that shows how to perform Principal Component Analysis (PCA) on the extracted features to identify patterns and reduce dimensionality.



## The mindmetrix_biomarker package
### Data loader module
The `loader.py` module is responsible for loading the datasets.

### Data Preprocessing module
The `preprocess` module contains functions to preprocess the physiological and subjects data. You can use these functions to clean and prepare your data for feature extraction. 

For example, you can preprocess your physiological data using the `preprocess_physiological_data` function:

```python
from mindmetrix_biomarker.preprocess import preprocess_physiological_data
# Example usage
data = preprocess_physiological_data('path_to_your_data.csv')
```

### Features extraction module
The `extract` module contains functions to extract features from the preprocessed data. You can use the `extract_features` function to extract relevant features for your analysis:

```python
from mindmetrix_biomarker.exctract import extract_features
# Example usage
features = extract_features(preprocessed_data)
```







