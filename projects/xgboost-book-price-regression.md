# Book Price Prediction with XGBoost Regression

This project was an experiment, using XGBoost and a publicly available dataset to predict the price of a book given some characteristics.
The main intention of this project was to explore machine learning techniques, including data cleaning and model explainability.

## Motivations

### Data Source

The data for this project was sourced from [Kaggle](https://www.kaggle.com/datasets/die9origephit/amazon-data-science-books), a repository of publicly available datasets for machine learning and exploration.
It comes in the format of a comma-separated values (CSV) file, and is a table which contains information about listings for 830 different books on Data Science that are being sold on Amazon.

CSV files are easy to work with, as they are a plain-text file format, meaning that their raw content is human-readable with just a text editor.
The dataset was primarily chosen due to the amount of data cleaning that it would require to be in a suitable state for model training.
Many of the columns in the dataset contain values that should be separated into their own columns through a process called feature extraction.
Feature extraction is the process of transforming data points from their raw source format into a format which is compatible with a machine learning model ([Source: Snowflake](https://www.snowflake.com/guides/feature-extraction-machine-learning/)).
An example of this would be the `dimensions` column, which contains the width, depth, height, and associated measurement unit respectively of each book.
This provides an opportunity to separate each of the dimensions into columns of their own.

This dataset also provides a hyperlink to each book's Amazon listing, in the `complete_link` column.
This means that it is suitable for enrichment, which is a form of feature engineering.
Feature engineering is an umbrella term for preparing data for a machine learning model, and as such encompasses both enrichment and feature extraction.
Enrichment differs from feature extraction in that instead of features of the raw data being transformed, they are instead used as a form of key to help join on features from external sources ([Source: Alteryx](https://www.alteryx.com/glossary/data-enrichment)).
In this case, enrichment would involve scraping and processing data from the provided listing link to use as features.

### Why XGBoost?

XGBoost, short for Extreme Gradient Boosting, is an open-source machine learning algorithm based on the concept of decision trees.
In short, it is a highly efficient and accurate algorithm ([Source: Armand Ruiz](https://www.nocode.ai/why-xgboost-is-so-popular-among-data-scientists/)) for multiple machine learning tasks, including regression and classification, that has gained significant popularity over recent years.
This means that there are many resources available online to refer to when using almost any of its capabilities.
Furthermore, the fact that it is open-source means that its developers and contributors are oftentimes also its users.
This has lead to high-quality official documentation, and an easy-to-use Python API.

## Implementation

### File Structure

The first stage of any project is to create the file structure.
For this project, the following file structure was chosen:

```
.
└── project/
    ├── price_model/
    │   ├── __init__.py
    │   ├── _model.py
    │   └── _features/
    │       ├── __init__.py
    │       ├── _feature_extraction.py
    │       ├── _null_processing.py
    │       └── _feature_engineering.py
    └── main.py
```

It was a conscious decision to follow the name conventions of Python projects.
For example, `main.py` is commonly used to designate the main script, which is to be run by the user.
Additionally, naming specific module files with a leading underscore `_` is a hint to the user that they are private, and should not be imported from directly.
The public classes, methods, and attributes of each submodule are accessible instead from their `__init__.py` script.

### Code

#### Feature extraction

The dataset is loaded into the project through a package called Pandas.
Pandas is an open-source library for handling and processing tabular data, and is among the most popular libraries used by data scientists ([Source: DataCamp](https://www.datacamp.com/blog/top-python-libraries-for-data-science)).
Like XGBoost, it was chosen for its stability, documentation, and ease-of-use.

The first stage of processing is to extract and normalise features that were combined in the original data.
The most important column to format correctly is the the target column.
