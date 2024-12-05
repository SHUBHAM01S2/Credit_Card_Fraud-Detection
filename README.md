# Credit_Card_Fraud-Detection
💳 Credit Card Fraud Detection
This project is part of the Machine Learning with Python course by Coursera and IBM. It focuses on detecting fraudulent credit card transactions using various machine learning techniques and tools. The dataset used is sourced from Kaggle's Credit Card Fraud Detection.

🚀 Project Overview
Credit card fraud detection is a critical task in financial industries to prevent unauthorized transactions. This project demonstrates how machine learning can be applied to recognize fraudulent transactions effectively and efficiently.

🛠️ Technologies & Libraries Used
Programming Language: Python
Libraries:
Data Manipulation: Pandas, NumPy
Machine Learning Models: scikit-learn
Visualization: Matplotlib
Preprocessing: StandardScaler, normalize
📊 Dataset Details
Source: Credit Card Fraud Detection on Kaggle
Description:
284,807 rows representing credit card transactions.
31 columns, including the target variable Class, where:
0 = Legitimate transaction.
1 = Fraudulent transaction.
🔑 Key Features
Data Preprocessing:

Standardized numerical features.
Normalized data using L1 normalization.
Exploratory Data Analysis:

Distribution of transaction amounts and their range.
Imbalance analysis of target classes.
Model Building:

Decision Tree Classifier: Trained and evaluated using both scikit-learn and Snap ML to compare training speeds and accuracy.
Support Vector Machine (SVM): Implemented to handle class imbalance effectively using balanced weights.
Model Evaluation:

Used metrics like ROC-AUC score, hinge loss, and training speed to evaluate and compare models.

📈 Results
Training Speed: Snap ML outperformed scikit-learn, achieving a significant speed-up in model training.
Model Performance:
Decision Tree Classifier: Achieved a high ROC-AUC score for fraud detection.
Support Vector Machine: Effective in handling imbalanced datasets.
📚 Learnings from the Project
Handling imbalanced datasets using sampling techniques and balanced weights.
Preprocessing data for better machine learning model performance.

🙌 Acknowledgments
Special thanks to Coursera, IBM, and Kaggle for providing the resources and dataset for this project.

