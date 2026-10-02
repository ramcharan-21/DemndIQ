import requests
import sys

BASE_URL = "http://127.0.0.1:8000"
DASHBOARD_URL = "http://localhost:8501"
SKU = "SKU_000"

def test_api_root():
    try:
        resp = requests.get(f"{BASE_URL}/")
        if resp.status_code == 200:
            print("[OK] API Root is reachable.")
        else:
            print(f"[FAIL] API Root returned {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] API Root failed: {e}")

def test_inventory_endpoint():
    try:
        url = f"{BASE_URL}/inventory/{SKU}"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            if data["sku"] == SKU and "available_inventory" in data:
                print(f"[OK] Inventory Endpoint works for {SKU}. Available: {data['available_inventory']}")
            else:
                print(f"[FAIL] Inventory Endpoint returned unexpected data: {data}")
        else:
            print(f"[FAIL] Inventory Endpoint returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[FAIL] Inventory Endpoint failed: {e}")

def test_forecast_endpoint():
    try:
        url = f"{BASE_URL}/forecast/{SKU}"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            if "forecasted_demand" in data and "reorder_point" in data:
                print(f"[OK] Forecast Endpoint works for {SKU}. Demand: {data['forecasted_demand']}, Reorder Point: {data['reorder_point']}")
            else:
                print(f"[FAIL] Forecast Endpoint returned unexpected data: {data}")
        else:
            print(f"[FAIL] Forecast Endpoint returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[FAIL] Forecast Endpoint failed: {e}")

def check_dashboard():
    try:
        resp = requests.get(DASHBOARD_URL)
        if resp.status_code == 200:
            print("[OK] Dashboard is reachable.")
        else:
            print(f"[FAIL] Dashboard returned {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] Dashboard unreachable: {e}")

if __name__ == "__main__":
    print("Starting System Integration Verification...\n")
    test_api_root()
    test_inventory_endpoint()
    test_forecast_endpoint()
    check_dashboard()
    print("\nVerification Complete.")
