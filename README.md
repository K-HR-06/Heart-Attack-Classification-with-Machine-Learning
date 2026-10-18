# Heart Attack Classification with Machine Learning

A Machine Learning coursework project that compares classification algorithms for predicting the binary `heart_attack` label using demographic, lifestyle, and health-related variables.

The project includes data exploration, preprocessing, feature engineering, hyperparameter tuning, and comparisons of individual models and ensemble methods.

## Dataset

`heart_attack_prediction_indonesia.csv` contains **158,355 records** and **28 columns**, including the target variable.

Features include age, blood pressure, cholesterol measurements, smoking status, physical activity, and medical history.

The supplied files do not document the original dataset source or collection method.

## Methods

- Explore distributions and relationships using charts and a correlation heatmap.
- Encode categorical variables.
- Create two additional features: `cholesterol_ratio` and `avg_blood_pressure`.
- Train Logistic Regression, Decision Tree, and Random Forest models.
- Compare soft voting, hard voting, bagging, AdaBoost, and Gradient Boosting.
- Use randomized search for hyperparameter tuning.
- Evaluate accuracy, precision, recall, F1-score, confusion matrices, and ROC curves.

The final model comparison uses a **70% training and 30% testing split**, with **47,507 test records**.

## Recorded Results

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 72.36% |
| Tuned Logistic Regression | 73.12% |
| Decision Tree | 65.75% |
| Tuned Decision Tree | 73.52% |
| Random Forest | 73.20% |
| Tuned Random Forest | 73.62% |
| Soft Voting | 65.62% |
| Hard Voting | 70.58% |
| Bagging with Logistic Regression | 72.38% |
| Bagging with Decision Tree | 72.68% |
| AdaBoost with Logistic Regression | 73.13% |
| AdaBoost with Decision Tree | 65.57% |
| Tuned AdaBoost with Decision Tree | 65.57% |
| Gradient Boosting | **73.73%** |

Gradient Boosting achieved the highest recorded accuracy. For the positive class (`heart_attack = 1`), its saved classification report shows **0.69 precision**, **0.63 recall**, and **0.66 F1-score**.

These results come from saved notebook outputs and have not been independently reproduced.

## Tools

Python, pandas, NumPy, Matplotlib, seaborn, SciPy, and scikit-learn.

## Project Files

| File | Description |
| --- | --- |
| `code_AoL.ipynb` | Data exploration, preprocessing, model training, and evaluation |
| `heart_attack_prediction_indonesia.csv` | Dataset used in the analysis |

## How to Run

Open `code_AoL.ipynb` in Jupyter Notebook or Google Colab with the required libraries installed. Place the CSV in the working directory and run the cells in order.

The hyperparameter searches include many model fits and may take substantial time.

## Notes

- The CSV contains an alcohol-consumption category named `None`. The notebook reads it as missing and randomly replaces it with `Moderate` or `High`, altering the original category information.
- An earlier preprocessing section uses an 80/20 split, scaling, and outlier removal. The final models use a separate 70/30 split built from `df_fe`, so those earlier transformations are not applied to the final model inputs.
- Saved outputs include Logistic Regression convergence warnings. Random imputation also has no fixed NumPy seed, so rerun results may differ.

## Authors

Computer Science and Statistics, BINUS University

- Ann A. Halim
- A. Izzatil Isma
- Carmenita Angelica
- Marcell Prasetyo
