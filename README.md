# AI-ML-Assignments

# 🩺 Diabetes Prediction Model using Random Forest

This project demonstrates how to build a machine learning pipeline to predict the likelihood of diabetes using the Diabetes Prediction Dataset. It includes data preprocessing, model training, evaluation, and hyperparameter tuning using `RandomForestClassifier`.

---

## 📁 Dataset

We used the **Diabetes Prediction Dataset** which includes health-related attributes such as age, BMI, HbA1c levels, glucose levels, and more. The target variable is a binary classification (`0` or `1`), indicating the absence or presence of diabetes.

---

## 📌 Project Steps

### 1. 📦 Data Loading & Exploration

- Loaded the dataset using `pandas`.
- Explored the structure, data types, and missing values.
- Displayed the first few records using `df.head()`.
- Visualized distributions and correlation matrix using `matplotlib` and `seaborn`.

---

### 2. 🧹 Preprocessing

- **Handled Missing Values**: Checked and handled any null values.
- **Encoding**: Applied one-hot encoding on categorical features like `gender` and `smoking_history`.
- **Scaling**: Applied `StandardScaler` on numerical features to normalize their values.
- **Splitting**: Split the dataset into training and testing sets using `train_test_split`.

---

### 3. 🤖 Model Training

- Used `RandomForestClassifier` with default parameters to train on the scaled training data.
- Trained the model using:
  
  ```python
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)


### 4.🧾 Requirements

1. Python 3.10+
2. pandas
3. numpy
4. matplotlib
5. seaborn
6. scikit-learn
7. joblib
