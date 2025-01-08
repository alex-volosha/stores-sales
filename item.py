import requests

url = 'http://localhost:9696/predict'

item = {
    'price': 19.99,
    'month': 3,
    'day_of_week': 2,
    'dept_name': 'ВОДКА,НАСТОЙКИ',
    'class_name': 'ВОДКА',
    'store_format': 'Format-1',
    'city': 'City1',
    'store_id': '3',
    'item_id': 'e4c10ab64623',
    'is_promo': 0,
    'is_markdown': 0,
    'is_weekend': 0,
    'is_month_end': 0,
    'price_change_percentage': -145.5,
    'date': '2025-03-15'

}

response = requests.post(url, json=item).json()
print(response)