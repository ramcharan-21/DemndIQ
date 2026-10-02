import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

try:
    from api import get_custom_summary
    
    # Payload that triggers the fallback (less than 2 rows or no history)
    payload = [{
        "date": "2023-01-01",
        "qty": 10,
        "sku": "fallback_test_sku"
    }]
    
    print("Testing get_custom_summary with minimal payload...")
    result = get_custom_summary(payload)
    
    print("\nResults:")
    print(json.dumps(result, indent=2))
    
    # Assertions to verify the fallback
    for item in result:
        if item["sku"] == "fallback_test_sku":
            assert item["forecasted_demand"] == 2, f"Expected 2, got {item['forecasted_demand']}"
            assert item["reorder_point"] == 2, f"Expected 2, got {item['reorder_point']}"
            assert item["suggested_order_qty"] == 2, f"Expected 2, got {item['suggested_order_qty']}"
    
    print("\nVerification Successful: Static fallback values are correctly applied.")

except Exception as e:
    print(f"\nVerification Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
