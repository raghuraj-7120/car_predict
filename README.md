# Car Price Prediction App

This project is a **Car Price Prediction App** that predicts the selling price of a car based on its specifications. It consists of two main components:

1. **Machine Learning Backend** - A Flask API built with a RandomForestRegressor model.
2. **Android App (Jetpack Compose with Kotlin)** - A user-friendly interface for predicting car prices using the deployed API.

---

## 🚀 Features
- Real-time price prediction of used cars.
- Clean and user-friendly UI built with Jetpack Compose.
- Backend API for prediction hosted on Render: [API Endpoint](https://car-predict-cqb9.onrender.com/predict)

---

## 📂 Project Structure
📦Car Price Prediction
├── app
│ ├── src
│ │ ├── main
│ │ │ ├── java/com/example/mycarpriceapp
│ │ │ │ └── MainActivity.kt
│ │ │ └── res
│ │ │ └── layout
│ │ │ └── activity_main.xml
├── backend
│ ├── cardata.csv
│ ├── car_price_model.pkl
│ ├── car_model_columns.pkl
│ ├── app.py
├── README.md
└── requirements.txt

---

## 🔍 Backend (Flask API)

The backend is a Flask API that takes car details as input and returns the predicted selling price. The model is trained using a **RandomForestRegressor** on a dataset of car specifications and prices.

### 📝 Model Training

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# Load the dataset
df = pd.read_csv('cardata.csv')

# Data Preprocessing
df.columns = df.columns.str.strip()
df.rename(columns={'vehicle_age': 'Car_Age'}, inplace=True)
df.drop(['Unnamed: 0', 'car_name', 'brand', 'model'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Features and Target variable
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the Model and Columns
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('car_model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model trained and saved successfully.")

