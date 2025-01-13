import duckdb
import zipfile
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge


# paramiters
n_estimators=100
learning_rate=0.1
max_depth=14

output_file = 'model.bin'
output_encoders = 'label_encoders.pkl'

# data preperation 

# Connect to DuckDB
con = duckdb.connect()

# Create simpler dataset with basic features
data = con.execute("""
    WITH sales_data AS (
    SELECT 
        s.date,
        s.store_id,
        s.item_id,
        CAST(s.quantity AS FLOAT) as quantity,
        CAST(s.price_base AS FLOAT) as price,
        CAST(CASE WHEN d.doc_id IS NOT NULL THEN 1 ELSE 0 END AS INTEGER) as is_promo,
        CAST(CASE WHEN m.price IS NOT NULL THEN 1 ELSE 0 END AS INTEGER) as is_markdown,
        CAST(COALESCE((d.sale_price_before_promo - d.sale_price_time_promo) / 
            NULLIF(d.sale_price_before_promo, 0) * 100, 0) AS FLOAT) as promo_discount,
        CAST(COALESCE((m.normal_price - m.price) / 
            NULLIF(m.normal_price, 0) * 100, 0) AS FLOAT) as markdown_percentage,
        CAST(COALESCE((ph.price - s.price_base) / 
            NULLIF(s.price_base, 0) * 100, 0) AS FLOAT) as price_change_percentage,
        EXTRACT(MONTH FROM s.date) as month,
        EXTRACT(DOW FROM s.date) as day_of_week,
        c.dept_name,
        c.class_name,
        st.format as store_format,
        st.city
    FROM read_csv_auto('ml-zoomcamp-2024-competition/sales.csv') s
    LEFT JOIN read_csv_auto('ml-zoomcamp-2024-competition/catalog.csv') c ON s.item_id = c.item_id
    LEFT JOIN read_csv_auto('ml-zoomcamp-2024-competition/stores.csv') st ON s.store_id = st.store_id
    LEFT JOIN read_csv_auto('ml-zoomcamp-2024-competition/discounts_history.csv') d 
        ON s.item_id = d.item_id 
        AND s.date = d.date
        AND s.store_id = d.store_id
    LEFT JOIN read_csv_auto('ml-zoomcamp-2024-competition/markdowns.csv') m 
        ON s.item_id = m.item_id 
        AND s.date = m.date
        AND s.store_id = m.store_id
    LEFT JOIN read_csv_auto('ml-zoomcamp-2024-competition/price_history_cleaned.csv') ph
        ON s.item_id = ph.item_id
        AND s.date = ph.date 
        AND s.store_id = ph.store_id
    )
    SELECT *
    FROM sales_data
""").fetchdf()

# Remove negative prices
data = data[data['price'] > 0]

data['price'] = np.log1p(data['price'])

model = Ridge(alpha=1.0)

# Add seasonal features
data['is_weekend'] = (data['day_of_week'].isin([5,6])).astype(int)
data['is_month_end'] = (pd.to_datetime(data['date']).dt.is_month_end).astype(int)
data['promo_price_ratio'] = data['price'] / data.groupby(['item_id'])['price'].transform('mean')

# Prepare features
def prepare_features(data):
    # Encode categorical variables
    cat_columns = ['dept_name', 'class_name', 'store_format', 'city','item_id','store_id','is_promo','is_markdown']
    label_encoders = {}
    
    for col in cat_columns:
        data[col] = data[col].fillna('Unknown')  # Handle missing values
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    # Fill missing numerical values
    data = data.fillna(0)
    
    return data, label_encoders

# Prepare the data
data, label_encoders = prepare_features(data)

# Split features and target
target = 'quantity'
features = ['price', 'month', 'day_of_week', 'dept_name', 'class_name', 
            'store_format', 'city', 'store_id', 'item_id','is_promo','is_markdown',
            'is_weekend','is_month_end','price_change_percentage']

X = data[features]
y = data[target]

# Split train/test based on date
split_date = '2024-06-01'  # adjust based on your data
train_mask = pd.to_datetime(data['date']) < split_date
X_train = X[train_mask]
X_test = X[~train_mask]
y_train = y[train_mask]
y_test = y[~train_mask]


# Train model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=14,
    random_state=42,
    #enable_categorical=True
)
model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation Metrics:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")


# save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)
print(f'The model is saved to {output_file}')

#Save the label encoders
with open(output_encoders, 'wb') as f_out:
    pickle.dump(label_encoders, f_out)
print(f'The label encoders are saved to {output_encoders}')
