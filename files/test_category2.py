import urllib.request, json, pandas as pd

API_URL = "http://127.0.0.1:8001"

data = {
    "transaction_date": ["2026-03-01", "2026-03-02", "2026-03-03"],
    "Product_Category": ["Beauty", "Clothing", "Books"],
    "Purchase_amount": [100.0, 150.0, 120.0]
}
df = pd.DataFrame(data)
payload = df.to_dict(orient="records")
req = urllib.request.Request(
    f"{API_URL}/dashboard/custom_summary",
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

with urllib.request.urlopen(req) as resp:
    print('Status:', resp.status)
    result = json.load(resp)
    for item in result:
        print('SKU:', item.get('sku'), 'Forecast:', item.get('forecasted_demand'))
