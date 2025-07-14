import pandas as pd
from flask import Flask, request, jsonify
import pickle


# Load the saved model
with open('california_knn_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize app
app = Flask(__name__)

@app.route('/')
def home():
    return "👋 Server is running. Use POST on /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON and convert to DataFrame
        data = request.get_json()
        input_df = pd.DataFrame([data])  # wrap in list for a single row

        # Predict
        prediction = model.predict(input_df)[0]

        # Return prediction
        return jsonify({'predicted_house_value': round(float(prediction), 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Running this took forever the first time, so I commented it and ran in cmd
if __name__ == '__main__':
    app.run(debug=True)
