import requests
try:
    r = requests.get("http://127.0.0.1:8000/dashboard/summary", timeout=20)
    print(r.status_code)
    if r.status_code == 200:
        print(r.json()[:2]) # Print first 2 items
    else:
        print(r.text)
except Exception as e:
    print(e)
