import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import google.generativeai as genai

# New imports for forecasting
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini-Powered Inventory & Forecasting",
    page_icon="📦",
    layout="wide"
)

# --- Gemini API Configuration ---
def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API. Error: {e}")
        return False

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith(('.xls','xlsx')):
        return pd.read_excel(uploaded_file)
    else:
        return None

def get_gemini_response(prompt, model_name="gemini-2.0-flash"):
    try:
        model = genai.GenerativeModel(model_name)
        return model.generate_content(prompt).text
    except Exception as e:
        return f"An error occurred with Gemini: {e}"

@st.cache_data(show_spinner=False)
def forecast_demand(df, date_col, quantity_col, periods=30):
    # Prepare data for Prophet
    ts = df[[date_col, quantity_col]].rename(columns={date_col:'ds', quantity_col:'y'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(ts)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast

# --- Main App ---
st.title("📦 Gemini-Powered Inventory & Forecasting Suite")

# Sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Google Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload inventory (CSV/Excel)", type=["csv","xls","xlsx"])
    st.markdown("---")
    st.info("Required columns: Product, Quantity, Price, Area, Last Sale Date")

if not api_key:
    st.warning("Enter your Gemini API Key to proceed.")
    st.stop()
if not configure_gemini(api_key):
    st.stop()
if not uploaded_file:
    st.info("Upload your inventory file to continue.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.error("Unsupported file type.")
    st.stop()

# Column Mapping
with st.expander("✏️ Map Your Columns", expanded=True):
    columns = ["-"] + df.columns.tolist()
    product_col = st.selectbox("Product Col", columns, index=1)
    qty_col     = st.selectbox("Quantity Col", columns, index=2)
    price_col   = st.selectbox("Unit Price Col", columns, index=3)
    area_col    = st.selectbox("Area Col", columns, index=4)
    date_col    = st.selectbox("Last Sale Date Col", columns, index=5)

required = [product_col, qty_col, price_col, area_col]
if "-" in required:
    st.error("Map all required columns.")
    st.stop()

# Preprocess
df[qty_col]   = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
df['Total Value'] = df[qty_col] * df[price_col]
if date_col != "-":
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['Days Since Last Sale'] = (datetime.now() - df[date_col]).dt.days

# Define tabs (now 6)
tabs = st.tabs([
    "📊 Dashboard",
    "🗺️ Area-wise Charts",
    "📈 Demand Forecasting",
    "🚚 Last-Mile Transfers & Sustainability",
    "📉 Deadstock Prediction",
    "💡 Deadstock to Live"
])

# --- Tab 1: Dashboard ---
with tabs[0]:
    st.header("Overview")
    total_units = int(df[qty_col].sum())
    total_value = df['Total Value'].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Units", f"{total_units:,}")
    k2.metric("Inventory Value", f"${total_value:,.2f}")
    k3.metric("Unique SKUs", df[product_col].nunique())
    k4.metric("Areas", df[area_col].nunique())

    st.subheader("AI Summary")
    prompt = f"""
Analyze this inventory and give 3-5 bullet points on health, risks & key opportunities.
- SKUs: {df[product_col].nunique()}
- Units: {total_units}
- Value: ${total_value:,.2f}
"""
    with st.spinner("Generating summary…"):
        st.markdown(get_gemini_response(prompt))

    st.markdown("---")
    st.dataframe(df)

# --- Tab 2: Area-wise Charts ---
with tabs[1]:
    st.header("Inventory by Area")
    agg = df.groupby(area_col)['Total Value'].sum().reset_index()
    chart_type = st.radio("Chart type", ["Bar","Pie"], horizontal=True)
    if chart_type=="Bar":
        fig = px.bar(agg, x=area_col, y='Total Value', text_auto=".2s")
    else:
        fig = px.pie(agg, names=area_col, values='Total Value', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Demand Forecasting ---
with tabs[2]:
    st.header("Demand Forecasting")
    if date_col == "-":
        st.warning("Map a Last Sale Date column to forecast.")
    else:
        st.info("Forecast next 30 days total units sold (all SKUs).")
        # sum by day
        daily = df.groupby(date_col)[qty_col].sum().reset_index()
        m, forecast = forecast_demand(daily, date_col, qty_col, periods=30)
        fig1 = plot_plotly(m, forecast)  # interactive plotly
        st.plotly_chart(fig1, use_container_width=True)

        upcoming = forecast[['ds','yhat']].tail(30)
        avg_pred = upcoming['yhat'].mean()
        st.metric("Avg. Daily Forecast (30d)", f"{avg_pred:,.0f}")

# --- Tab 4: Last-Mile Transfers & Sustainability ---
with tabs[3]:
    st.header("Transfer Suggestions & Sustainability KPIs")
    # 1) Compute forecast vs. current stock per area (using mean daily forecast proportionately)
    st.info("We’ll allocate forecast demand by area based on current inventory distribution.")
    area_qty = df.groupby(area_col)[qty_col].sum().reset_index()
    total_qty = area_qty[qty_col].sum()
    # assume each area's daily demand = (area stock / total stock) * avg_daily_forecast
    if date_col != "-":
        avg_forecast = avg_pred
    else:
        avg_forecast = df[qty_col].sum()/30
    area_qty['Est. Daily Demand'] = area_qty[qty_col]/total_qty * avg_forecast
    area_qty['Surplus/Deficit'] = area_qty[qty_col] - (area_qty['Est. Daily Demand']*7)  # 7-day demand
    st.dataframe(area_qty.sort_values('Surplus/Deficit'))

    # 2) Suggest transfers: move from surplus to deficit
    surplus = area_qty[area_qty['Surplus/Deficit']>0].copy()
    deficit = area_qty[area_qty['Surplus/Deficit']<0].copy()
    st.subheader("Suggested Transfers (Top 3)")
    transfers = []
    for _, d in deficit.iterrows():
        # pick largest surplus area
        s = surplus.sort_values('Surplus/Deficit', ascending=False).iloc[0]
        qty_move = min(s['Surplus/Deficit'], abs(d['Surplus/Deficit']))
        transfers.append({
            "From": s[area_col],
            "To": d[area_col],
            "Qty": int(qty_move)
        })
        surplus.loc[surplus[area_col]==s[area_col],'Surplus/Deficit'] -= qty_move
    st.table(pd.DataFrame(transfers))

    # 3) Sustainability & Cost KPIs
    CO2_PER_MILE_LB = 1.5   # lb CO2 per mile (example)
    COST_PER_MILE = 2.00    # $ per mile
    AVG_DISTANCE = 100      # miles average transfer
    total_moves = sum(t['Qty'] for t in transfers)
    est_miles = total_moves * AVG_DISTANCE / 50  # assume 50 units per truck
    co2_saved = est_miles * CO2_PER_MILE_LB
    cost_est = est_miles * COST_PER_MILE

    st.subheader("Sustainability & Cost Estimates")
    c1, c2, c3 = st.columns(3)
    c1.metric("Units to Move", f"{total_moves}")
    c2.metric("Est. CO₂ Emissions (lb)", f"{co2_saved:,.0f}")
    c3.metric("Est. Transport Cost", f"${cost_est:,.2f}")

    # 4) Event Simulation
    st.subheader("📋 Event Simulation")
    scenario = st.selectbox("Pick a scenario", [
        "Local Festival ➜ boost demand", 
        "Snowstorm ➜ block delivery", 
        "Flash Sale ➜ surge orders"
    ])
    if st.button("Generate Adaptive Plan"):
        prompt = f"""
You are a supply-chain strategist. A '{scenario}' is about to happen.
Given current inventory distribution and forecast, suggest:
1) Actions to rebalance stock 
2) Last-mile delivery adjustments
3) Communication strategy
Present as bullet points.
"""
        st.markdown(get_gemini_response(prompt))

# --- Tab 5: Deadstock Prediction ---
with tabs[4]:
    st.header("Deadstock Identification")
    if date_col=="-":
        st.warning("Map Last Sale Date to proceed.")
    else:
        thresh = st.slider("Days since last sale≥", 30, 730, 180, 15)
        dead = df[df['Days Since Last Sale']>thresh]
        if dead.empty:
            st.success("No deadstock!")
        else:
            st.error(f"{len(dead)} SKUs totalling ${dead['Total Value'].sum():,.2f}")
            st.dataframe(dead[[product_col, qty_col,'Total Value','Days Since Last Sale',area_col]])
            st.session_state.deadstock_df = dead

# --- Tab 6: Deadstock to Live ---
with tabs[5]:
    st.header("Deadstock → Live Strategies")
    if 'deadstock_df' not in st.session_state or st.session_state.deadstock_df.empty:
        st.info("Run Deadstock Prediction first.")
    else:
        if st.button("Generate Plan"):
            dead = st.session_state.deadstock_df
            sample = dead.head(20)[[product_col, qty_col,'Total Value','Days Since Last Sale']]
            prompt = f"""
You are an expert inventory strategist. Create a 'Deadstock to Live' plan for these items:
{sample.to_string(index=False)}
Total deadstock value: ${dead['Total Value'].sum():,.2f}
Structure your response in Markdown:
1. Triage & Prioritization
2. ≥4 Sales/Marketing Strategies (with concrete examples)
3. Internal Process Improvements
4. Final Recommendation
"""
            plan = get_gemini_response(prompt)
            st.markdown(plan)