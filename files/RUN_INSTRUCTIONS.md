# How to Run BizChainConnect

The BizChainConnect project consists of two primary services that must be running simultaneously to work correctly: a FastAPI backend and a Streamlit frontend dashboard.

## Prerequisites

Before starting the application, ensure you are in the project root directory and have your virtual environment activated.

```powershell
# Navigate to the project directory
cd c:\Users\anilk\bizchainconnect

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

---

## Step 1: Start the API Backend
The backend API handles all the machine learning model inferences and data processing operations.

1. Open a new terminal or PowerShell window.
2. Ensure you are in the project root containing your code.
3. Start the FastAPI server by running the following command:

```bash
# Wait for the server to say "Application startup complete"
python -m uvicorn src.api:app --port 8001 --reload
```

*Note: If the `python` command defaults to a global installation, use the explicit virtual environment path instead:*
```bash
.\venv\Scripts\python.exe -m uvicorn src.api:app --port 8001 --reload
```

This will start the API at `http://localhost:8001`. You can view the interactive API documentation and test endpoints at `http://localhost:8001/docs`.

---

## Step 2: Start the Streamlit Dashboard
The dashboard provides an interactive user interface to view inventory states, stockouts, and aggregated demand forecasts.

1. Open a **second** (separate) terminal or PowerShell window.
2. Make sure you are also in the project root directory.
3. Start the Streamlit dashboard by running the following command:

```bash
python -m streamlit run src/dashboard.py
```

*Note: If the `streamlit` or `python` command defaults to a global installation, use the explicit virtual environment path instead:*
```bash
.\venv\Scripts\python.exe -m streamlit run src/dashboard.py
```

This command will automatically open the dashboard in your default web browser, typically located at `http://localhost:8501`.

---

## Stopping the Application
To shut down the application, return to each terminal window running the services and press `Ctrl + C` on your keyboard. This will safely stop the server instances.
