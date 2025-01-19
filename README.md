# IIT

This repository includes code to train models to predict treatment interruption. Data was collected via KenyaEMR and centralized in a National Data Warehouse. No data is included in this repository, but may be made available upon request with necessary approvals. 

The repository includes the following scripts:

1) Sampling - sample 300K patient IDs from the national data warehouse and extract their clinical visit, pharmacy, lab, and demographic extracts.
2) IITPrep - compute outcomes for each observations and generate input features.
3) DriftAnalysis - explores trends in outcome variable, establishing statistically significant downward trend.
4) Mice - impute missing values using Multiple Imputations with Chained Equations
5) Model training scripts
   a) Catboost
   b) RF_CV
   c) XGB
   d) LogisticRegression
 6) BestPerformingModel - analysis and explanation of best performing model.

For more information, please contact Jonathan Friedman at Jonathan.Friedman@thepalladiumgroup.com
