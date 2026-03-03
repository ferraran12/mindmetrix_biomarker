### Getting Started mindmetrix_biomarker package

To use the mindmetrix_biomarker package, follow the instructions below to set up your environment and install the package:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -e .
```

### Data Preprocessing
The `preprocess` module contains functions to preprocess the physiological and subjects data. You can use these functions to clean and prepare your data for feature extraction. For example, you can preprocess your physiological data using the `preprocess_physiological_data` function:

```python
from mindmetrix_biomarker.preprocess import preprocess_physiological_data
# Example usage
data = preprocess_physiological_data('path_to_your_data.csv')
```


### Features extraction
The `extract` module contains functions to extract features from the preprocessed data. You can use the `extract_features` function to extract relevant features for your analysis:

```python
from mindmetrix_biomarker.exctract import extract_features
# Example usage
features = extract_features(preprocessed_data)
```


### Results
In the `results` directory, you can find three Jupyter notebooks: 
- `exploration.ipynb` that provides an overview of the dataset and its characteristics,
- `biomarkers.ipynb` that shows the extracted features in relation with the STAI. 
- `PCA.ipynb` that shows how to perform Principal Component Analysis (PCA) on the extracted features to identify patterns and reduce dimensionality.







