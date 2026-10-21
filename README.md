# ECE607-Defense-Contract-ML
Machine learning analysis of positive contract value growth in U.S. Department of Defense construction awards.
# Predicting Positive Contract Value Growth in U.S. Defense Construction Awards Using Machine Learning

**Author:** Antoine Davis  
**Institution:** Old Dominion University  
**Course:** ECE 607  
**Date:** October 2026

## Project Overview

This project investigates whether information available at the time of a U.S. Department of Defense construction contract award can predict subsequent positive changes in recorded contract value.

Using an FPDS-derived dataset containing 360,029 transaction records, I cleaned and organized the data into contract-level observations, resulting in 17,647 candidate contract groups used for machine learning analysis.

## Machine Learning Models

I evaluated three classifiers:

1. Logistic Regression
2. Gaussian Naive Bayes
3. Class-Weighted Logistic Regression

The models used original award value, award year, and contract type to predict a preliminary binary indicator of positive contract value growth.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 65.21% | 61.02% | 42.70% | 50.24% |
| Gaussian Naive Bayes | 58.81% | 25.00% | 0.07% | 0.14% |
| Balanced Logistic Regression | 64.28% | 55.91% | 62.19% | 58.88% |

Balanced logistic regression improved positive-case detection but produced more false-positive predictions.

## Dataset

The data were obtained from a publicly available Federal Procurement Data System (FPDS)-derived construction contract dataset hosted on Mendeley Data.

Source: https://data.mendeley.com/datasets/yk4s7pdsvk/1

The dataset is not included in this repository. Users should obtain the source data separately.

## How to Reproduce the Results

1. Download the project notebook from this repository.
2. Open the notebook in Google Colab.
3. Download the source dataset from Mendeley Data.
4. Extract the dataset and locate the CSV file.
5. Rename the CSV file `FPDSData.csv` if necessary.
6. Upload the CSV into the Google Colab `/content/` directory.
7. Install or confirm the required Python dependencies.
8. Select **Runtime → Run all** to execute the notebook.
9. Verify that the analysis produces 17,647 final observations and a stratified split of 14,117 training and 3,530 testing observations.
10. Compare the output accuracy, precision, recall, F1 scores, and confusion matrices with the reported results.

## Software Requirements

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Limitations

The outcome represents calculated positive contract value growth based on modification records. It has not been independently validated as a true construction cost overrun.

Financial field interpretations, differences in contract observation periods, and contract identification assumptions require additional validation before using the model for operational predictions.

## Conclusion

This project demonstrates an end-to-end supervised machine learning workflow for analyzing Department of Defense construction contract records. The findings show that class weighting improved positive-case recall, while Gaussian naive Bayes performed poorly in identifying positive cases.
