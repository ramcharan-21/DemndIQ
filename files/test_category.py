import urllib.request
import urllib.parse
import pandas as pd
import json

API_URL = "http://127.0.0.1:8001"

# Create a test dataset mirroring "Product Category" behavior
data = {
    "transaction_date": ["2026-03-01", "2026-03-02", "2026-03-03"],
    "Product_Category": ["Beauty", "Clothing", "Books"],
    "Purchase_amount": [100.0, 150.0, 120.0]
}
df = pd.DataFrame(data)
payload = df.to_dict(orient="records")

req_data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(f"{API_URL}/dashboard/custom_summary", data=req_data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        res_body = response.read()
        
        result = json.loads(res_body)
        print("Predictions returned:")
        for item in result:
             print(f"- SKU Identified: {item['sku']}")
             if item["sku"] == "Global_SKU":
                  print("FAILED: Fallback to Global_SKU occurred instead of category.")
        
except urllib.error.URLError as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print(e.read().decode())
