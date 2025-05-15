# Car Price Prediction App

This project is a **Car Price Prediction App** that predicts the selling price of a car based on its specifications. It consists of two main components:

1. **Machine Learning Backend** - A Flask API built with a RandomForestRegressor model.
2. **Android App (Jetpack Compose with Kotlin)** - A user-friendly interface for predicting car prices using the deployed API.

---

## 🚀 Features
- Real-time price prediction of used cars.
- Clean and user-friendly UI built with Jetpack Compose.
- Backend API for prediction hosted on Render: [API Endpoint](https://car-predict-cqb9.onrender.com/predict)

```

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

```

## 🔍 Backend (Flask API)

The backend is a Flask API that takes car details as input and returns the predicted selling price. The model is trained using a **RandomForestRegressor** on a dataset of car specifications and prices.
``` python

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Car Price Prediction API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lazy loading of the model and column names
        with open('car_price_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('car_model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)

        # JSON input
        input_data = request.get_json()

        # Validate input
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Create input array in correct column order
        input_array = [0] * len(model_columns)
        for i, col in enumerate(model_columns):
            input_array[i] = input_data.get(col, 0)

        input_array = np.array(input_array).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)
        
        return jsonify({'predicted_price': round(prediction[0], 2)})

    except FileNotFoundError as fnf_error:
        return jsonify({'error': f'Model file not found: {fnf_error}'}), 500

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

```

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

```
📱 Android App (Jetpack Compose)
The Android application is built with Jetpack Compose and communicates with the backend to predict car prices based on user inputs.

🔨 Key Technologies:
Kotlin

Jetpack Compose

Retrofit for API Calls

Material3 Design

🔧 Setup
Open the project in Android Studio.

Add Internet permission in AndroidManifest.xml:
<uses-permission android:name="android.permission.INTERNET" />
Run the app on an emulator or physical device.

🛠️ Usage
Enter the car details (Brand, Model, Age, etc.).

Click the Predict Price button.

Wait for the API to respond and display the predicted price.

🌐 API Reference
POST /predict

Request Body (JSON):
```
{
  "brand": "Maruti",
  "model": "Swift",
  "vehicle_age": 3,
  "km_driven": 40000,
  "seller_type": "Individual",
  "fuel_type": "Petrol",
  "transmission_type": "Manual",
  "mileage": 20.0,
  "engine": 1197,
  "max_power": 82.0,
  "seats": 5
}
```
Response (JSON):
```
{
  "predicted_price": 350000
}
```

Author
Build by Raghuraj , Dhruv and Sarthak
For any issues or inquiries, please contact me at [raghuraj7210@gmail.com].
