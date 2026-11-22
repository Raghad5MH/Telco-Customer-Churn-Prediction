# Telco Customer Churn Prediction with a Simple ANN

## Project Overview
This project uses customer information from a telecommunications company to predict whether a customer is likely to leave the company (churn) or stay. It is a **binary classification** problem built around a simple Artificial Neural Network (ANN).

The project also compares the ANN with a Logistic Regression baseline and explores regularization and hyperparameter tuning.

## Dataset
- **Source:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers and 21 columns.
- **Target:** `Churn`
  - `Yes`: the customer left.
  - `No`: the customer stayed.

The dataset includes customer details, account information, subscribed services, contract type, and charges.

## Project Workflow

1. **Data Understanding**
   - Load the dataset and inspect its structure and columns.
   - Check data types, missing values, blank values, and duplicates.
   - Identify `customerID` as an identifier rather than a useful prediction feature.

2. **Exploratory Data Analysis (EDA)**
   - Explore numerical and categorical features.
   - Study the distribution of the target variable.
   - Compare churn rates across customer groups.
   - Review correlations between selected numerical features.

3. **Data Preprocessing**
   - Convert `TotalCharges` to a numeric column.
   - Replace its 11 blank values with `0`. These rows belong to customers with `tenure = 0`.
   - Remove `customerID`.
   - Encode categorical variables using binary encoding and one-hot encoding.
   - Split the data into training, validation, and test sets using a 60/20/20 split.
   - Apply `StandardScaler`, fitting it on the training set only.

4. **Model Building**
   - Train Logistic Regression as a baseline.
   - Build an ANN with two hidden layers.
   - Use ReLU activation in the hidden layers and Sigmoid in the output layer.
   - Train with the Adam optimizer and Binary Crossentropy loss.

5. **Regularization and Hyperparameter Tuning**
   - Experiment with Dropout, L2 regularization, and Early Stopping.
   - Compare different network sizes and Dropout values.
   - Select the best ANN using validation ROC-AUC.

6. **Evaluation**
   - Evaluate the selected ANN and Logistic Regression on the test set.
   - Use Accuracy, Precision, Recall, F1-score, ROC-AUC, a confusion matrix, and an ROC curve.

7. **Model Saving and Inference**
   - Save the trained ANN, scaler, and feature-column list.
   - Load the saved files and check that predictions remain consistent.
   - Use the `predict_churn()` function to make predictions for a new customer.

## Results

The best ANN was selected using the validation set. The final test-set results were:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.789 | 0.619 | 0.527 | 0.569 | **0.832** |
| ANN - Dropout + EarlyStop | 0.776 | 0.604 | 0.452 | 0.517 | 0.823 |

Logistic Regression performed slightly better than the selected ANN on the test set. This is an important finding: a more complex neural network does not always outperform a simpler model.

During hyperparameter tuning, the best validation ROC-AUC was **0.856**, using 32 and 16 neurons with a Dropout rate of 0.4. This validation result is separate from the final test result above.

## Key EDA Findings

The analysis showed higher churn rates among customers who:
- Have month-to-month contracts.
- Have a shorter tenure with the company.
- Use fiber optic internet.
- Pay by electronic check.
- Do not have Online Security or Tech Support services.

Gender showed little difference in churn in this dataset.

## Project Files

```text
Telco-Churn-Project/
├── Telco_Churn_Deep_Learning_Project.ipynb
├── README.md
├── best_model.keras          # Generated when the notebook is run
├── scaler.pkl                # Generated when the notebook is run
└── feature_columns.pkl       # Generated when the notebook is run
```

The three model files are created by the notebook's model-saving section. They do not need to be present before running the notebook.

## Requirements

The notebook uses Python and the following main libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow / Keras
- kagglehub
- joblib

Install the packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow kagglehub joblib
```

## How to Run

1. Clone or download this repository.
2. Install the required packages.
3. Open `Telco_Churn_Deep_Learning_Project.ipynb` in Jupyter Notebook or Google Colab.
4. Run the notebook cells in order.
5. The dataset is downloaded from Kaggle through `kagglehub`. Make sure your Kaggle access is configured in the environment if authentication is requested.
6. The notebook saves the trained model and supporting preprocessing files in the working directory.

## Limitations and Future Work

- The target classes are imbalanced: approximately one quarter of customers churn.
- Recall can be improved to catch more customers who are likely to leave.
- Future experiments could use class weights or a lower prediction threshold.
- Other models, such as Random Forest and XGBoost, could be compared.
- Cross-validation could provide a more reliable estimate of model performance.

## Team
- Raghad Alharbi.
- Ruba Alotaibi.
