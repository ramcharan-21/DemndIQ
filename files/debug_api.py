import sys
import os
from pathlib import Path
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from api import get_forecast
    print("Testing get_forecast('SKU_000') directly...")
    result = get_forecast("SKU_000")
    print("Result:", result)
except Exception:
    traceback.print_exc()

import requests
try:
    print("\nTesting via HTTP...")
    r = requests.get("http://127.0.0.1:8000/forecast/SKU_000")
    print(f"Status: {r.status_code}")
    print("Content:", r.text)
except Exception as e:
    print("HTTP Request failed:", e)
