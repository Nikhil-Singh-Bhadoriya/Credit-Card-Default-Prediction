# Credit Card Default Prediction

A machine learning project to predict credit card default payments using classification algorithms.

## 🎯 Key Results

- **Best Model**: Gaussian Naive Bayes with `var_smoothing=0.5`
- **Accuracy**: 80.08%
- **Dataset**: 1,001 samples with 23 features
- **Performance Gain**: 10.36% improvement over baseline through hyperparameter tuning

## 📋 Project Overview

This project implements machine learning models to predict whether a credit card holder will default on their next payment. The analysis includes data exploration, preprocessing, model training, hyperparameter tuning, and evaluation.

## 🗂️ Dataset

- **File**: `creditCardFraud_28011964_120214.csv`
- **Samples**: 1,001 credit card holders
- **Features**: 23 features
- **Target Variable**: `default payment next month` (binary classification)
- **Feature Types**: Payment history, bill statements, payment amounts, and demographic information

## 🔧 Technologies Used

- **Python 3.14**
- **Libraries**:
  - pandas - Data manipulation and analysis
  - numpy - Numerical computations
  - scikit-learn - Machine learning algorithms and preprocessing
  - xgboost - Gradient boosting classifier
  - matplotlib - Data visualization

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

### Running the Project

1. Clone the repository
2. Ensure the dataset file `creditCardFraud_28011964_120214.csv` is in the project directory
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook credit_card_fault_detection.ipynb
   ```
4. Run all cells sequentially

## 📊 Project Workflow

### 1. Data Loading and Exploration
- Load dataset using pandas
- Display first 15 rows
- Check data types and information
- Identify missing values
- Generate statistical summary

### 2. Data Preprocessing
- **Feature Selection**: Separate features (X) and target variable (y)
- **Train-Test Split**: 75% training (751 samples), 25% testing (250 samples) with `random_state=50`
- **Feature Scaling**: StandardScaler normalization to standardize features for optimal model performance

### 3. Model Development

#### Naive Bayes Classifier (GaussianNB)
- **Baseline Model Accuracy**: 69.72%
- **Hyperparameter Tuning**: 
  - GridSearchCV with 5-fold cross-validation
  - Parameter tested: `var_smoothing` [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 0.001, 0.01, 0.05, 0.1, 0.5]
  - **Best Parameter**: `var_smoothing=0.5`
- **Optimized Model Accuracy**: 80.08% ✅ **(+10.36% improvement)**

#### XGBoost Classifier
- **Baseline Model Accuracy**: 77.29%
- **Hyperparameter Tuning**:
  - GridSearchCV with 5-fold cross-validation
  - Parameters tested:
    - `n_estimators`: [50, 100, 130]
    - `max_depth`: [3, 4, 5, 6, 7, 8, 9, 10]
    - `random_state`: [0, 50, 100]
  - **Best Parameters Found**: `max_depth=8`, `n_estimators=50`, `random_state=0`
- **Final Model Accuracy**: 77.29% (using custom parameters: `max_depth=4`, `n_estimators=90`, `random_state=0`)

### 4. Model Evaluation
- **Metric**: Accuracy Score
- **Best Performing Model**: Gaussian Naive Bayes with `var_smoothing=0.5` achieved **80.08% accuracy**
- GridSearchCV used for systematic hyperparameter optimization
- 5-fold cross-validation for robust model selection

## 📈 Model Performance Summary

| Model | Type | Baseline Accuracy | Optimized Accuracy | Improvement |
|-------|------|-------------------|-----------------------|-------------|
| Gaussian Naive Bayes | Probabilistic Classifier | 69.72% | 80.08% | +10.36% |
| XGBoost | Gradient Boosting | 77.29% | 77.29% | - |

## 🔍 Key Features

- **Data Quality Checks**: Missing value detection and data type verification
- **Feature Engineering**: Proper train-test split to prevent data leakage
- **Standardization**: Feature scaling for improved model performance
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Model Comparison**: Multiple algorithms evaluated

## 📝 Notes

- **Python 3.14 Compatibility**: This project uses Python 3.14, which is not compatible with `pandas-profiling` or `ydata-profiling`. Data profiling is performed using built-in pandas methods.
- **Data Leakage Prevention**: Train-test split is performed before scaling to avoid data leakage
- **Reproducibility**: Random state set to 50 for consistent results

## 📂 Project Structure

```
Credit-Card-Default-Prediction-main/
│
├── credit_card_fault_detection.ipynb    # Main Jupyter notebook
├── creditCardFraud_28011964_120214.csv  # Dataset
└── README.md                            # Project documentation
```

## 🎯 Results & Insights

- ✅ Successfully achieved **80.08% accuracy** with optimized Naive Bayes
- ✅ Hyperparameter tuning significantly improved Naive Bayes performance (+10.36%)
- ⚠️ XGBoost showed stable performance but no improvement with hyperparameter tuning
- 📊 Dataset contains 1,001 samples with 23 features for binary classification


