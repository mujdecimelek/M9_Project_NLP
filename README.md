# M9_Project_NLP
# Disaster Tweet Classification

This project classifies tweets as disaster-related (1) or not disaster-related (0) using machine learning techniques.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Results](#results)
7. [How to Use](#how-to-use)
8. [Contact](#contact)

## Overview
This project involves natural language processing (NLP) and machine learning to classify tweets. The goal is to accurately predict whether a given tweet refers to a real disaster.

## Dataset
The dataset contains the following columns:
- `id`: Unique identifier for each tweet.
- `text`: The content of the tweet.
- `keyword`: A keyword from the tweet (may contain missing values).
- `location`: The location where the tweet was sent (may contain missing values).
- `target`: The target label (1 = disaster, 0 = non-disaster).

The `train.csv` dataset is used for training, and the `test.csv` dataset is used for evaluation.

## Preprocessing
The preprocessing steps include:
1. Removing URLs, mentions, and special characters.
2. Converting text to lowercase.
3. Cleaning text data by removing unnecessary spaces.

## Feature Engineering
Additional features are created to improve the model performance:
- **Text Length**: Number of characters in the tweet.
- **Word Count**: Number of words in the tweet.
- **Emoji Count**: Number of emojis used in the tweet.
- **Uppercase Ratio**: Ratio of uppercase characters in the tweet.

Text data is converted into numerical format using **TF-IDF Vectorization**.

## Modeling
The following machine learning models were used:
1. Logistic Regression
2. Random Forest
3. Multinomial Naive Bayes
4. XGBoost

GridSearchCV was applied to optimize hyperparameters. Logistic Regression provided the best performance.

## Results
The optimized Logistic Regression model achieved the following metrics:
- **Accuracy**: 82%
- **F1-Score**: 77%
- **Precision**: 85%
- **Recall**: 71%

## How to Use
1. Run the preprocessing and feature engineering steps.
2. Train the optimized Logistic Regression model.
3. Predict the `target` values for the test dataset.
4. Save predictions to a `submission.csv` file.

### Requirements
The following Python libraries are required:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

### Run the Code
Execute the following command to run the project:

## Contact
For any queries or clarifications, please reach out via email or GitHub.

- **Your Name** :Melek Müjdeci and Handenur Kesat 
- **Email**: melek.mujdeci@amsterdamtech.com
- **GitHub**: 
