Customer Churn Analysis and Predictive Modeling
Project Overview

This project implements a comprehensive machine learning framework to analyze customer churn and develop predictive models for proactive retention strategies. Using a structured synthetic dataset of 50 customers, the study integrates multi-source customer data, performs advanced feature engineering, and applies supervised learning techniques to predict churn risk.

Multiple classification algorithms—including Logistic Regression, Random Forest, and XGBoost—are trained and evaluated using robust performance metrics such as ROC-AUC, Precision, Recall, and F1-Score. In addition, customer segmentation techniques are applied to identify high-risk and high-value cohorts, enabling targeted business interventions.

The project emphasizes both predictive accuracy and interpretability, providing actionable insights into key churn drivers.

Datasets

The analysis integrates five structured datasets:

customer_demographics.csv
Contains demographic attributes including age, gender, income, tenure, region, and household characteristics (50 records).

customer_transactions.csv
Includes transactional metrics such as total transactions, average order value, total spend, recency, and preferred product category (50 records).

customer_support.csv
Captures customer service interactions including support tickets, average resolution time, complaints, and preferred communication channel (50 records).

customer_survey.csv
Contains customer feedback metrics including product satisfaction, service satisfaction, price perception, Net Promoter Score (NPS), and recommendation score (50 records).

customer_churn.csv
Includes churn indicator (binary classification target) and churn date (50 records).

Analytical Objectives

Identify demographic, transactional, behavioral, and experiential factors influencing customer churn

Develop and compare supervised learning models for churn classification

Evaluate model performance using ROC-AUC, Precision, Recall, and F1-Score

Conduct feature importance analysis for model interpretability

Perform customer segmentation using clustering techniques

Define risk-value quadrants for strategic prioritization

Formulate data-driven retention strategies based on model insights

Methodology

Data Integration and Preprocessing

Dataset merging and consistency validation

Missing value handling

Categorical encoding

Feature scaling and normalization

Feature Engineering

Behavioral indicators (e.g., recency-frequency-monetary patterns)

Support intensity metrics

Satisfaction index aggregation

Churn propensity indicators

Model Development

Logistic Regression (baseline interpretable model)

Random Forest (ensemble-based model)

XGBoost (gradient boosting model)

Model Evaluation

ROC-AUC curve analysis

Confusion matrix analysis

Precision, Recall, F1-Score comparison

Cross-validation

Customer Segmentation

Clustering-based segmentation (e.g., K-Means)

Risk-value categorization

Retention strategy mapping

Installation

Install required dependencies using:

pip install -r requirements.txt
