from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model and column names
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('car_model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Car Price Prediction API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON input
        input_data = request.get_json()

        # Create input array in correct column order
        input_array = [0] * len(model_columns)
        for i, col in enumerate(model_columns):
            if col in input_data:
                input_array[i] = input_data[col]

        input_array = np.array(input_array).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)
        return jsonify({'predicted_price': round(prediction[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
