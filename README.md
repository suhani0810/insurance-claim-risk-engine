# Insurance Claim Prediction Risk Engine

## Project Objective
In the insurance industry, predicting which policyholders are likely to file a claim is essential for settting premiums. This project involved building a risk engine optimized for highly imbalanced and noisy data.

## Domain Analysis
The dataset features are organized into logical groupings (Binary, Categorical, and Continuous) to maintain policyholder anonymity:
  - **id/target:** Unique identifier and the claim indicator (1 = claim filed).
  - **ps_ind_ (Individual):** Features related to the individual policyholder (e.g., ps_ind_01 to ps_ind_18).
  - **ps_reg_ (Regional):** Data related to the geographical region of the policyholder; ps_reg_03 (population density) was a top predictor.
  - **ps_car_ (Vehicle):** Attributes of the insured vehicle; ps_car_13 (vehicle performance) was highly significant.
  - **ps_calc_ (Calculated):** Derived features (e.g., ps_calc_01 to ps_calc_20). My analysis proved these were "noise" and were removed to improve accuracy.
  - **Suffixes:**
    - **_cat:** Categorical features.
    - **_bin:** Binary features (0 or 1).
  - **No suffix:** Continuous or ordinal features.

## Key Insights
- **Clean Data Wins:** Removing "ps_calc" noise was foundational for model success.
- **Universal Risk Factors:** Vehicle performance (ps_car_13) and regional density (ps_reg_03) were identified as core risk indicators across all models.
- **Optimization:** Moving from a baseline LightGBM to a Tuned XGBoost provided a significant accuracy lift.

## Technical Highlights
- Utilized Bayesian hyperparameter tuning (Optuna) for model optimization.
- Achieved a final model metric of **ROC-AUC 0.6427**.

## Technologies Used
- XGBoost & LightGBM
- Optuna (Bayesian Optimization)
