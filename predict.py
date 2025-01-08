import pickle
from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd

# Load model and label encoders
model_file = 'model.bin'
encoders_file = 'label_encoders.pkl'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(encoders_file, 'rb') as f_in:
    label_encoders = pickle.load(f_in)

app = Flask('retail_forecast')

MODEL_FEATURES = [
    'price', 'month', 'day_of_week', 'dept_name', 'class_name', 
    'store_format', 'city', 'store_id', 'item_id', 'is_promo',
    'is_markdown', 'is_weekend', 'is_month_end', 'price_change_percentage'
]


def prepare_features(data):
    """Apply feature engineering and label encoding to the input DataFrame"""
    
    # Handle categorical variables
    cat_columns = ['dept_name', 'class_name', 'store_format', 'city', 
                  'item_id', 'store_id', 'is_promo', 'is_markdown']
    
    # Apply label encoding
    for col in cat_columns:
        data[col] = data[col].fillna('Unknown')  # Handle missing values
        # Transform using pre-fitted label encoders
        try:
            data[col] = label_encoders[col].transform(data[col])
        except KeyError as e:
            print(f"Warning: Unknown category in {col}. Using default category.")
            # Use the first category from training as default
            data[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])
    
    # Feature engineering
    data['is_weekend'] = (data['day_of_week'].isin([5,6])).astype(int)
    data['is_month_end'] = (pd.to_datetime(data['date']).dt.is_month_end).astype(int)
    data['promo_price_ratio'] = data['price'] / data.groupby(['item_id'])['price'].transform('mean')
    data['price_log'] = np.log1p(data['price'])
    
    data = data[MODEL_FEATURES]

    # Fill any remaining missing values
    data = data.fillna(0)
    
    return data

@app.route('/predict', methods=['POST'])
def predict():
    item = request.get_json()
    
    # Convert item to DataFrame
    data = pd.DataFrame([item])
    
    # Apply feature engineering and encoding
    data = prepare_features(data)
    
    # Make prediction
    y_pred = model.predict(data)
    
    result = {
        'Predicted quantity sold on that day': float(y_pred[0])  # Convert numpy float to Python float
    }

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 9696))
    app.run(host="0.0.0.0", port=port)