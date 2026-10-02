import streamlit as st
import pandas as pd
import requests
from pathlib import Path

# ---------------- Config ----------------
API_URL = "http://127.0.0.1:8001"
DATA_DIR = Path("data")
FEATURE_FILE = DATA_DIR / "features" / "Time series features_ml.csv"  # Ensure this path is correct

st.set_page_config(
    page_title="Autonomous Inventory Reorder",
    layout="wide"  # Changed to wide for better table view
)

st.title("📦 Autonomous Inventory Reorder System")
st.caption("AI-powered demand forecasting & autonomous reorder decisions")

# ---------------- Load SKUs ----------------
@st.cache_data
def load_skus():
    try:
        df = pd.read_csv(FEATURE_FILE)
        return sorted(df["sku"].unique().tolist())
    except:
        return []

skus = load_skus()

# ---------------- Simulation State ----------------
if 'simulated_inventory' not in st.session_state:
    st.session_state['simulated_inventory'] = {}

def get_data_with_simulation():
    """Fetches summary data and overrides with simulated inventory if present."""
    try:
        summary_resp = requests.get(f"{API_URL}/dashboard/summary", timeout=10)
        if summary_resp.status_code == 200:
            data = summary_resp.json()
            df = pd.DataFrame(data)
            
            # Apply Simulation overrides
            for index, row in df.iterrows():
                sku = row['sku']
                if sku in st.session_state['simulated_inventory']:
                    sim_val = st.session_state['simulated_inventory'][sku]
                    df.at[index, 'available_inventory'] = sim_val
                    
                    # Recalculate logic locally for simulation
                    reorder_point = row['reorder_point']
                    forecast = row['forecasted_demand']
                    
                    if forecast > 0:
                        df.at[index, 'days_of_supply'] = round(sim_val / forecast, 1)
                    else:
                        df.at[index, 'days_of_supply'] = 999.0

                    if sim_val <= reorder_point:
                        df.at[index, 'action'] = "REORDER"
                        df.at[index, 'stock_status'] = "LOW" # Simplified status update
                        target_stock = reorder_point + (forecast * 30)
                        df.at[index, 'suggested_order_qty'] = max(1, target_stock - sim_val)
                        df.at[index, 'recommendation_reason'] = "Inventory below Reorder Point (Sim)"
                    else:
                        df.at[index, 'action'] = "OK"
                        df.at[index, 'stock_status'] = "OK"
                        df.at[index, 'suggested_order_qty'] = 0
                        df.at[index, 'recommendation_reason'] = "Healthy Stock Level (Sim)"
                        
            return df
        else:
            st.error("Failed to load data.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

# ---------------- Navigation ----------------
page = st.sidebar.radio("Navigate", ["Actionable Insights", "Product Predictions", "Global Overview", "Custom Predictions"])

# ==========================================
# PAGE 1: ACTIONABLE INSIGHTS
# ==========================================
if page == "Actionable Insights":
    st.subheader("🚀 Actionable Insights & Recommendations")
    
    if st.button("Refresh Recommendations"):
        st.cache_data.clear()
        # Clear simulation on refresh if desired, or keep it. 
        # st.session_state['simulated_inventory'] = {} 

    df = get_data_with_simulation()
    
    if not df.empty:
        # Filter for reorders
        reorder_df = df[df["action"] == "REORDER"].copy()
        
        # KPI Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Critical Reorders Needed", len(reorder_df), delta_color="inverse")
        if not reorder_df.empty:
            total_qty_needed = reorder_df["suggested_order_qty"].sum()
            c2.metric("Total Units to Order", f"{int(total_qty_needed):,}")
        
        # Priority Table
        st.divider()
        if not reorder_df.empty:
            st.info("The following items have dropped below their reorder point. Suggested quantities target a 30-day supply.")
            
            # Format for display
            display_cols = [
                "sku", "available_inventory", "forecasted_demand", 
                "days_of_supply", "reorder_point", "suggested_order_qty", "recommendation_reason"
            ]
            
            st.dataframe(
                reorder_df[display_cols].sort_values("days_of_supply", ascending=True),
                column_config={
                    "days_of_supply": st.column_config.NumberColumn("Days Supply", format="%.1f days"),
                    "suggested_order_qty": st.column_config.NumberColumn("Suggested Qty (Units)"),
                },
                use_container_width=True,
                height=500
            )
        else:
            st.success("🎉 No immediate reorders needed! Inventory levels are healthy.")

# ==========================================
# PAGE 2: PRODUCT PREDICTIONS
# ==========================================
elif page == "Product Predictions":
    st.subheader("🔎 Single Product Deep Dive")
    
    if not skus:
        st.error("No SKUs found. Check data source.")
    else:
        # ---------------- UI Inputs (Side-by-Side) ----------------
        c_input1, c_input2 = st.columns(2)
        
        with c_input1:
            sku = st.selectbox("Select SKU", skus)

        # ---------------- Inventory Lookup ----------------
        # Fetch inventory early to populate defaults if not in session state
        current_inv_default = 0
        inventory_data = {}

        try:
            r = requests.get(f"{API_URL}/inventory/{sku}", timeout=5)
            if r.status_code == 200:
                inventory_data = r.json()
                current_inv_default = inventory_data.get("available_inventory", 0)
        except Exception:
            pass
        
        # Use session state if available, else default
        if sku in st.session_state['simulated_inventory']:
            current_inv_val = st.session_state['simulated_inventory'][sku]
        else:
            current_inv_val = current_inv_default

        with c_input2:
            def update_inventory():
                st.session_state['simulated_inventory'][sku] = st.session_state[f"inv_input_{sku}"]

            current_inventory = st.number_input(
                "Current Inventory Level (simulated)",
                min_value=0,
                value=int(current_inv_val),
                key=f"inv_input_{sku}",  # Unique key per SKU
                on_change=update_inventory
            )
            # Ensure session state is synced if user didn't trigger on_change (e.g. initial load)
            st.session_state['simulated_inventory'][sku] = current_inventory

        # ---------------- Inventory Status ----------------
        st.divider()
        st.subheader("📊 Inventory Status (Live from System)")

        if inventory_data:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Available Inventory ({sku})", f"{inventory_data['available_inventory']} units")
            c2.metric("On Hand", inventory_data["on_hand"])
            c3.metric("Reserved", inventory_data["reserved"])

            if inventory_data["status"] == "OUT_OF_STOCK":
                st.error("❌ Out of Stock")
            elif inventory_data["status"] == "LOW":
                st.warning("⚠️ Low Inventory")
            else:
                st.success("✅ Inventory Healthy")

            st.caption(f"Snapshot date: {inventory_data['snapshot_date']} | Last updated: {inventory_data['last_updated']}")
        else:
            st.error("Inventory data not found or API error")


        # ---------------- Reorder Logic ----------------
        st.divider()
        st.subheader("📦 Reorder Decision Logic")

        # Fetch forecasted demand dynamically from API
        try:
            forecast_resp = requests.get(f"{API_URL}/forecast/{sku}", timeout=5)
            if forecast_resp.status_code == 200:
                forecast_data = forecast_resp.json()
                forecasted_demand = forecast_data.get("forecasted_demand", 0.0)
                reorder_point = forecast_data.get("reorder_point", 0.0)
            else:
                forecasted_demand = 0.0
                reorder_point = 0.0
                st.warning("Forecast not available.")
        except Exception as e:
            forecasted_demand = 0.0
            reorder_point = 0.0
            st.error(f"Forecast API error: {e}")

        c1, c2 = st.columns(2)
        c1.metric("Forecasted Daily Demand", forecasted_demand)
        c2.metric("Reorder Point", reorder_point)

        # ---------------- Effective Inventory ----------------
        st.caption("Comparing 'Current Inventory Level' against 'Reorder Point'")
        effective_inventory = current_inventory
        st.metric("Effective Inventory Used", effective_inventory)

        # ---------------- Decision ----------------
        if effective_inventory <= reorder_point:
            st.error("🔴 Reorder Required")
        else:
            st.success("🟢 No Reorder Needed")

# ==========================================
# PAGE 3: GLOBAL OVERVIEW
# ==========================================
elif page == "Global Overview":
    st.subheader("🌍 Global Inventory Overview")
    
    if st.button("Refresh Data"):
        st.cache_data.clear()
        # st.session_state['simulated_inventory'] = {}

    df_summary = get_data_with_simulation()

    if not df_summary.empty:
        # 1. Metrics Logic
        total_skus = len(df_summary)
        reorder_needed = df_summary[df_summary["action"] == "REORDER"].shape[0]
        
        c1, c2 = st.columns(2)
        c1.metric("Total SKUs", total_skus)
        c2.metric("Reorder Alerts", reorder_needed, delta_color="inverse")

        # Search Bar
        st.divider()
        search_query = st.text_input("🔍 Search by SKU", "").strip().upper()

        if search_query:
            df_summary = df_summary[df_summary["sku"].str.contains(search_query, na=False)]

        # 2. Table
        st.dataframe(
            df_summary.style.map(
                lambda x: "color: red; font-weight: bold" if x == "REORDER" else "color: green", 
                subset=["action"]
            ),
            use_container_width=True,
            height=500
        )
        
        # 3. Graph (Inventory vs Reorder Point)
        if not df_summary.empty:
            st.subheader("Inventory vs Reorder Point")
            chart_data = df_summary[["sku", "available_inventory", "reorder_point"]].set_index("sku")
            st.bar_chart(chart_data)

# ==========================================
# PAGE 4: CUSTOM PREDICTIONS
# ==========================================
elif page == "Custom Predictions":
    st.subheader("📂 Custom Data Predictions")
    st.markdown("Upload a CSV containing SKU features to generate dynamic demand forecasts and autonomous reorder decisions.")

    uploaded_file = st.file_uploader("Upload Features CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_custom = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            with st.expander("Preview Data"):
                st.dataframe(df_custom.head())
                
            if st.button("Run Predictions"):
                with st.spinner("Generating insights..."):
                    
                    df_to_process = df_custom
                    if len(df_to_process) > 50000:
                        st.warning(f"⚠️ Your file is extremely large ({len(df_to_process)} rows). Truncating to the first 50,000 rows to prevent browser memory limits. The backend will aggregate this timeline by SKU and Date.")
                        df_to_process = df_to_process.head(50000)
                        
                    # Convert to list of dicts for JSON payload
                    payload = df_to_process.to_dict(orient="records")
                    
                    try:
                        resp = requests.post(f"{API_URL}/dashboard/custom_summary", json=payload, timeout=90)
                        if resp.status_code == 200:
                            data = resp.json()
                            df_result = pd.DataFrame(data)
                            
                            if not df_result.empty:
                                # 1. Metrics Logic
                                total_skus = len(df_result)
                                reorder_needed = df_result[df_result["action"] == "REORDER"].shape[0]
                                
                                c1, c2 = st.columns(2)
                                c1.metric("Total SKUs Processed", total_skus)
                                c2.metric("Reorder Alerts", reorder_needed, delta_color="inverse")

                                st.divider()
                                # 2. Results Table
                                st.subheader("Prediction Results")
                                st.dataframe(
                                    df_result.style.map(
                                        lambda x: "color: red; font-weight: bold" if x == "REORDER" else "color: green", 
                                        subset=["action"]
                                    ),
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # 3. Graph
                                st.subheader("Inventory vs Reorder Point (Custom Data)")
                                chart_data = df_result[["sku", "available_inventory", "reorder_point"]].set_index("sku")
                                st.bar_chart(chart_data)
                            else:
                                st.warning("No valid predictions returned.")
                        else:
                            st.error(f"API Error: {resp.status_code} - {resp.text}")
                    except Exception as e:
                        st.error(f"Failed to communicate with API: {e}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
