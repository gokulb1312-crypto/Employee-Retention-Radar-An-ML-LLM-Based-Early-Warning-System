# Part 2 — Supervised Machine Learning

Trains a baseline logistic-regression classifier and a cross-validated tuned random forest for employee attrition.

Step 1: Load Part 1's cleaned data. Attrition is the binary label; EmployeeID is excluded because it is an identifier, not an employee characteristic.
Step 2: Split data into stratified training (80%) and test (20%) sets. Stratification preserves the attrition rate in both sets.
Step 3: Build a preprocessing pipeline. Numeric values are median-imputed and scaled; categories are one-hot encoded. The pipeline prevents test-set leakage.
Step 4: Fit logistic regression as the transparent baseline, then tune a class-balanced random forest using five-fold cross-validation.
Step 5: Compare F1 and ROC-AUC. F1 is important here because attrition classes are often imbalanced; ROC-AUC checks ranking quality across thresholds.
Step 6: Save the higher held-out-F1 pipeline as attrition_model.joblib, plus metrics JSON, confusion matrix, ROC curve, and feature-importance plot. The dashboard loads this saved pipeline.

Run: python train_model.py

Outputs include `attrition_model.joblib`, `metrics.json`, a confusion matrix/ROC plot, and feature importances.