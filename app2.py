import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="📦 Inventory & Last-Mile Hub",
    page_icon="🚀",
    layout="wide"
)

# --- GLOBAL CSS & BRANDING ---
import streamlit as st
import plotly.io as pio
from streamlit_option_menu import option_menu

# 1. Dark theme everywhere
pio.templates.default = "plotly_dark"

# 2. Global CSS for dark mode, cards, headers, and pill nav
st.markdown("""
<style>
  :root {
    --bg: #121212;
    --fg: #f2f8fd;
    --card-bg: #007dc6;
    --primary: #ffc120;
    --accent: #ffc120;
  }
  .stApp { background:var(--bg); color:var(--fg); }
  /* Top bar */
  .top-nav { position:sticky;top:0;z-index:999;
    background:var(--card-bg); padding:.75rem 1rem;
    display:flex;align-items:center;justify-content:space-between;
    border-bottom:1px solid #333;
  }
  .top-nav .title { color:var(--primary); font-size:1.8rem; font-weight:600; }
  .top-nav .help-btn {
    background:var(--accent); color:var(--bg);
    border:none; padding:.5rem 1rem; border-radius:4px;
    font-size:0.9rem; cursor:pointer;
  }
  /* Section cards */
  .metric-card {
    background:var(--card-bg); border-left:4px solid var(--accent);
    padding:1rem; border-radius:6px; margin-bottom:1rem;
    box-shadow:0 2px 6px rgba(0,0,0,0.7);
  }
  .metric-card h4 { margin:0; color:var(--primary); font-size:0.85rem; }
  .metric-card p  { margin:.3rem 0 0; font-size:1.6rem; color:var(--fg); }
  /* Pill menu override - hide default radio */
  .css-1aumxhk { visibility:hidden; height:0; }
</style>
""", unsafe_allow_html=True)

# --- TOP NAV BAR ---
st.markdown("""
<div class="top-nav">
  <div class="title">📦 Inventory & Last-Mile Hub</div>
  <button class="help-btn" onclick="window.open('https://your-docs-link','_blank')">
    ❓ Help
  </button>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- SIDEBAR: SETUP + NAVIGATION ---
def show_metric_card(col, title, val, suffix=""):
  col.markdown(f"""
    <div class="metric-card">
      <h4>{title}</h4>
      <p>{val}{suffix}</p>
    </div>
  """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Setup")
    api_key = st.text_input("🔑 Gemini API Key", type="password")
    upload  = st.file_uploader("📂 Inventory File", type=["csv","xlsx","xls"])
    
    st.markdown("---")

page = page if 'page' in locals() else "Dashboard"

# --- PREREQUISITE CHECKS ---
if not api_key:
    st.warning("🔑 Please enter your Gemini API Key.")
    st.stop()

# Configure Gemini
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Gemini: {e}")
    st.stop()

if not upload:
    st.info("📂 Please upload your inventory file to continue.")
    st.stop()

@st.cache_data
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xls","xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        return None

# Load DataFrame
df = load_data(upload)
if df is None:
    st.error("❌ Unsupported file format.")
    st.stop()

# --- COLUMN MAPPING ---
with st.expander("✏️ Map Columns", expanded=False):
    cols = ["–"] + list(df.columns)
    prod_col  = st.selectbox("Product ▶", cols, index=1)
    qty_col   = st.selectbox("Quantity ▶", cols, index=2)
    price_col = st.selectbox("Unit Price ▶", cols, index=3)
    area_col  = st.selectbox("Area/Location ▶", cols, index=4)
    date_col  = st.selectbox("Last Sale Date ▶", cols, index=5)
    lat_col = st.selectbox("Latitude (optional)", cols, index=0, help="If you have geo-coordinates")
    lon_col = st.selectbox("Longitude (optional)", cols, index=0, help="If you have geo-coordinates")

# --- NAVIGATION PILL MENU ---
if df is not None:
  page = option_menu(
    menu_title=None,  # no title
    options=["Dashboard","Area Charts","Forecast","Last-Mile","Deadstock","Restock"],
    icons=["bar-chart","geo-alt","graph-up","truck","file-earmark-excel","lightbulb"],
    default_index=0,
    orientation="horizontal",
    styles={
      "container": {"padding":"0!important", "background":":var(--card-bg)"},
      "nav-link": {
        "font-size":"1rem", "color":"var(--fg)", "padding":"0.6rem 1.2rem",
        "margin":"0 0.25rem", "border-radius":"6px"
      },
      "nav-link-selected": {
        "background-color":"var(--accent)", "color":"var(--bg)"
      }
  })
    
required = [prod_col, qty_col, price_col, area_col]
if "–" in required:
    st.error("🔴 Please map all required columns.")
    st.stop()

# --- PREPROCESSING ---
df[qty_col]   = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
df["TotalValue"] = df[qty_col] * df[price_col]
if date_col != "–":
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["DaysSinceSale"] = (datetime.now() - df[date_col]).dt.days

# --- UTIL: Gemini Prompt Wrapper ---
def get_gemini(text_prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(text_prompt).text
    except Exception as e:
        return f"💥 Gemini API Error: {e}"
    
@st.cache_data
def forecast_prophet(df, ds_col, y_col, periods=30):
    ts = df[[ds_col, y_col]].rename(columns={ds_col: "ds", y_col: "y"})
    m = Prophet(daily_seasonality=True)
    m.fit(ts)
    future = m.make_future_dataframe(periods=periods)
    fc = m.predict(future)
    return m, fc

from geopy.geocoders import Nominatim
import math

@st.cache_data
def geocode_areas(area_list):
    """Resolve area names into lat/lon via Nominatim (OpenStreetMap)."""
    geolocator = Nominatim(user_agent="last-mile-app", timeout=10)
    records = []
    for area in area_list:
        try:
            loc = geolocator.geocode(area)
            if loc:
                records.append({
                    "area": area,
                    "lat": loc.latitude,
                    "lon": loc.longitude
                })
        except Exception:
            continue
    return pd.DataFrame(records)

def haversine(lat1, lon1, lat2, lon2):
    """Compute distance (miles) between two lat/lon points."""
    R = 3958.8  # Earth radius in miles
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# --- PAGE FUNCTIONS ---


def page_dashboard():
    with st.container():
        st.header("📊 Dashboard: Inventory Overview")

        # Compute KPIs
        total_units = int(df[qty_col].sum())
        total_val   = df["TotalValue"].sum()
        sku_count   = df[prod_col].nunique()
        area_count  = df[area_col].nunique()

        # Display metric cards
        c1, c2, c3, c4 = st.columns(4, gap="small")
        show_metric_card(c1, "Total Units", f"{total_units:,}")
        show_metric_card(c2, "Inventory Value", f"${total_val:,.2f}")
        show_metric_card(c3, "Unique SKUs", sku_count)
        show_metric_card(c4, "Number of Areas", area_count)
        st.divider()
        st.subheader("🔍 AI-Powered Insight")
        prompt = (
            f"Inventory status: SKUs={sku_count}, Units={total_units}, Value=${total_val:,.2f}. "
            "Provide 3 concise bullet points on health, risks, and opportunities."
        )
        with st.spinner("Analyzing with Gemini…"):
            insight = get_gemini(prompt)
        st.markdown(insight)

        st.divider()
        st.subheader("🗒️ Data Preview")
        st.dataframe(df.head(15), use_container_width=True)

# Stub out other pages for now
# def page_area_charts(): st.header("🗺️ Area Charts (Coming Soon)")
def page_forecast():     st.header("📈 Forecast (Coming Soon)")
def page_last_mile():    st.header("🚚 Last-Mile (Coming Soon)")
def page_deadstock():    st.header("📉 Deadstock (Coming Soon)")
def page_restock():      st.header("💡 Restock (Coming Soon)")

def page_area_charts():
    with st.container():
        st.header("🗺️ Inventory Distribution by Area")
        st.markdown("Visualize how your inventory is spread across your locations.")

        # 1) Aggregate data
        value_agg = df.groupby(area_col)["TotalValue"].sum().reset_index()
        units_agg = df.groupby(area_col)[qty_col].sum().reset_index()

        # 2) User selects metric & chart style
        metric = st.selectbox("Metric", ["Total Value ($)", "Total Units"])
        style  = st.radio("Chart Type", ["Bar Chart", "Pie Chart"], horizontal=True)

        if metric == "Total Value ($)":
            data = value_agg.sort_values("TotalValue", ascending=False)
            y_col = "TotalValue"
            y_label = "Value ($)"
            colorscale = px.colors.sequential.Blues
        else:
            data = units_agg.rename(columns={qty_col: "TotalUnits"}).sort_values("TotalUnits", ascending=False)
            y_col = "TotalUnits"
            y_label = "Units"
            colorscale = px.colors.sequential.Teal

        # 3) Render chart
        if style == "Bar Chart":
            fig = px.bar(
                data,
                x=area_col,
                y=y_col,
                text=y_col,
                color=y_col,
                color_continuous_scale=colorscale,
                labels={area_col: "Area", y_col: y_label},
                title=f"{metric} by Area"
            )
        else:
            fig = px.pie(
                data,
                names=area_col,
                values=y_col,
                hole=0.4,
                color_discrete_sequence=colorscale,
                title=f"{metric} Distribution"
            )

        st.plotly_chart(fig, use_container_width=True)

        # 4) Optional AI Insight
        st.subheader("🔍 AI Insights")
        insight_prompt = (
            f"Inventory distribution by area for metric '{metric}':\n\n"
            f"{data.to_string(index=False)}\n\n"
            "Provide 3 bullet-point observations and 2 actionable recommendations."
        )
        if st.button("Generate AI Analysis"):
            with st.spinner("🧠 Working with Gemini…"):
                analysis = get_gemini(insight_prompt)
                st.markdown(analysis)

def page_forecast():
    with st.container():
        st.header("📈 Demand Forecasting")
        st.markdown("Use historical sales data to predict future demand and plan inventory accordingly.")

        # Guard if no date column mapped
        if date_col == "–":
            st.warning("⚠️ Please map a **Last Sale Date** column to enable forecasting.")
            return

        # 1) Forecast parameters
        with st.expander("⚙️ Forecast Settings", expanded=True):
            periods = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=7)
            daily_sea  = st.checkbox("Daily seasonality", value=True)
            weekly_sea = st.checkbox("Weekly seasonality", value=True)
            yearly_sea = st.checkbox("Yearly seasonality", value=True)

        # 2) Aggregate historical daily sales
        daily = df.groupby(date_col)[qty_col].sum().reset_index()
        daily = daily.rename(columns={date_col: "ds", qty_col: "y"})
        st.info(f"Using {len(daily)} days of history to forecast the next {periods} days.")

        # 3) Run Prophet (cached)
        m, fc = forecast_prophet(
            daily, 
            ds_col="ds", 
            y_col="y", 
            periods=periods,
            # pass seasonality flags through if you refactor forecast_prophet to accept them
        )

        # 4) Plot interactive forecast
        fig = plot_plotly(m, fc)
        st.plotly_chart(fig, use_container_width=True, height=500)

        # 5) Key forecast metrics
        future = fc.tail(periods)[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Forecast"})
        avg_forecast = future["Forecast"].mean()
        peak_row     = future.loc[future["Forecast"].idxmax()]
        peak_val     = int(peak_row["Forecast"])
        peak_date    = peak_row["Date"].date()

        c1, c2 = st.columns(2)
        c1.metric("Avg. Daily Demand", f"{avg_forecast:,.0f} units")
        c2.metric("Peak Demand", f"{peak_val:,} units on {peak_date}")

        # 6) Table of next 7 days
        st.subheader("Next 7 Days Forecast")
        st.dataframe(future.head(7), use_container_width=True)

        # 7) AI-Powered Demand Insight
        with st.expander("🤖 Generate AI Demand Analysis"):
            prompt = (
                f"Here are the last 30 days of daily sales:\n"
                f"{daily.tail(30).to_string(index=False)}\n\n"
                f"Here is the forecast for the next {periods} days (showing first 7):\n"
                f"{future.head(7).to_string(index=False)}\n\n"
                "Please provide:\n"
                "- Three observations on upcoming demand trends\n"
                "- Two actionable recommendations for inventory adjustments\n"
            )
            if st.button("🧠 Analyze with Gemini"):
                with st.spinner("Contacting Gemini…"):
                    insight = get_gemini(prompt)
                    st.markdown(insight)
            # ── AFTER your existing plots & metrics in page_forecast() ──
        st.divider()
        with st.expander("⚠️ Per-SKU/Area Risk Flags", expanded=False):
            grp_dim = st.radio("Group by", ["Product (SKU)", "Area"], horizontal=True)
            top_n   = st.number_input("Top N groups to analyze", min_value=3, max_value=20, value=5, step=1)
            horizon = st.slider("Horizon (days) for risk calc", 7, 30, 14, step=7)

            # 1) Build historical series per group
            if grp_dim == "Product (SKU)":
                col = prod_col 
            else:
                col = area_col

            # 2) pick top N by current stock
            top_groups = (
                df.groupby(col)[qty_col].sum()
                .sort_values(ascending=False)
                .head(top_n)
                .index
                .tolist()
            )

            risks = []
            for g in top_groups:
                sub = df[df[col] == g].groupby(date_col)[qty_col].sum().reset_index()
                sub = sub.rename(columns={date_col: "ds", qty_col: "y"})
                try:
                    m, fcast = forecast_prophet(sub, ds_col="ds", y_col="y", periods=horizon)
                    future_sum = fcast.tail(horizon)["yhat"].sum()
                except Exception:
                    # fallback: use mean*days
                    future_sum = sub["y"].mean() * horizon

                current_stock = df[df[col] == g][qty_col].sum()
                if future_sum > current_stock:
                    status = "🔴 Understock"
                elif future_sum < current_stock * 0.5:
                    status = "🟢 Overstock"
                else:
                    status = "🟡 Balanced"
                risks.append({
                    col: g,
                    "CurrentStock": int(current_stock),
                    f"Forecast{horizon}d": int(future_sum),
                    "Status": status
                })

            risk_df = pd.DataFrame(risks)
            st.table(risk_df)


def page_last_mile():
    with st.container():
            st.header("🚚 Last-Mile Transfers & Sustainability")
            st.markdown("Plan transfers & compare transport modes, with or without geo-data.")

            if date_col == "–":
                st.warning("Map a **Last Sale Date** column to estimate demand.")
                return

            # 1) Surplus/Deficit calc (7-day forecast)
            area_stock = (
                df.groupby(area_col)[qty_col]
                .sum()
                .reset_index(name="Stock")
            )
            total_stock = area_stock["Stock"].sum()

            daily = (
                df.groupby(date_col)[qty_col]
                .sum()
                .reset_index()
                .rename(columns={date_col: "ds", qty_col: "y"})
            )
            _, fc7 = forecast_prophet(daily, ds_col="ds", y_col="y", periods=7)
            avg7 = fc7.tail(7)["yhat"].mean()

            area_stock["EstDemand7d"] = area_stock["Stock"] / total_stock * (avg7 * 7)
            area_stock["SurplusDeficit"] = area_stock["Stock"] - area_stock["EstDemand7d"]

            # 2) Auto-geocode fallback
            coords_available = (lat_col != "–" and lon_col != "–")
            has_cached_geo = "geo_df" in st.session_state

            if not coords_available and not has_cached_geo:
                st.info("No geo-columns provided. You can auto-geocode your area names.")
                if st.button("📍 Auto-geocode Areas"):
                    areas = area_stock[area_col].unique().tolist()
                    geo_df = geocode_areas(areas)
                    if geo_df.empty:
                        st.error("Geocoding failed. Showing fallback chart.")
                    else:
                        st.success("Coordinates resolved! Rerun this page to see the map.")
                        st.session_state.geo_df = geo_df

                # Fallback: bar chart of surplus/deficit
                st.subheader("Surplus/Deficit by Area (Fallback)")
                st.bar_chart(
                    area_stock.set_index(area_col)["SurplusDeficit"],
                    height=300
                )
                return  # skip full map & planner UI

            # 3) Merge geo-data for map
            if has_cached_geo:
                geo = (
                    st.session_state.geo_df
                    .rename(columns={"area": area_col})
                    .merge(area_stock, on=area_col, how="inner")
                )
            else:
                geo = (
                    df[[area_col, lat_col, lon_col]]
                    .drop_duplicates(subset=[area_col])
                    .merge(area_stock, on=area_col, how="inner")
                    .rename(columns={lat_col: "lat", lon_col: "lon"})
                )

            # 4) Display map
            st.subheader("Map: Surplus/Deficit by Area")
            import plotly.express as px
            fig = px.scatter_mapbox(
                geo, lat="lat", lon="lon",
                color="SurplusDeficit", size="Stock",
                color_continuous_midpoint=0,
                color_continuous_scale=["red","white","green"],
                size_max=20, zoom=4,
                mapbox_style="open-street-map",
                hover_name=area_col,
                hover_data=["Stock","EstDemand7d","SurplusDeficit"],
            )
            st.plotly_chart(fig, use_container_width=True, height=400)

            st.divider()
                # ── INSIDE page_last_mile(), after computing area_stock & geo ──
            st.divider()
            with st.expander("🤖 Auto-Optimize Batch Schedule", expanded=False):
                st.markdown(
                    "Automatically assign each deficit area to its nearest surplus area."
                )
                if st.button("⚙️ Auto-Schedule Transfers"):
                    auto_plan = []
                    surplus_df = area_stock.query("SurplusDeficit>0").copy()
                    deficit_df = area_stock.query("SurplusDeficit<0").copy()

                    for _, d in deficit_df.iterrows():
                        # find closest surplus
                        distances = surplus_df.apply(
                            lambda s: haversine(s["lat"], s["lon"], d["lat"], d["lon"]), axis=1
                        )
                        idx = distances.idxmin()
                        s = surplus_df.loc[idx]
                        qty = min(s["SurplusDeficit"], abs(d["SurplusDeficit"]))
                        auto_plan.append({
                            "From": s[area_col],
                            "To":   d[area_col],
                            "Qty":  int(qty),
                            "Dist(mi)": round(distances.min(),1)
                        })
                        # update local surplus
                        surplus_df.at[idx, "SurplusDeficit"] -= qty

                    st.session_state["auto_batch"] = auto_plan
                    st.success("Auto-schedule generated!")

                if "auto_batch" in st.session_state:
                    st.table(pd.DataFrame(st.session_state["auto_batch"]))
                    if st.button("➕ Add Auto-Batch to Manual Plan"):
                        for r in st.session_state["auto_batch"]:
                            # reuse your existing batch structure
                            st.session_state["transfers"].append({
                                **r,
                                "Mode": "🚛 Van",  # default mode
                                "CO₂(lb)": "",
                                "Cost($)": ""
                            })
                        st.success("Auto-batch added to your manual plan!")

            # 5) Transfer Planner Controls
            st.subheader("📝 Plan a Transfer")
            col1, col2 = st.columns([2,1])
            surplus_areas = area_stock.query("SurplusDeficit>0")[area_col].tolist()
            deficit_areas = area_stock.query("SurplusDeficit<0")[area_col].tolist()

            with col1:
                from_area = st.selectbox("From (Surplus)", surplus_areas)
                to_area   = st.selectbox("To (Deficit)", deficit_areas)
                max_qty = int(min(
                    area_stock.query(f"{area_col}==@from_area")["SurplusDeficit"].iloc[0],
                    abs(area_stock.query(f"{area_col}==@to_area")["SurplusDeficit"].iloc[0])
                ))
                qty_move = st.slider("Quantity to Move", 1, max_qty, 1)

            with col2:
                st.markdown("**Transport Mode**")
                mode = st.radio("", ["🚚 Truck","🚛 Van","🚁 Drone"], index=1)
                profiles = {
                    "🚚 Truck": {"co2":3.0, "cost":1.8},
                    "🚛 Van":   {"co2":2.0, "cost":1.2},
                    "🚁 Drone": {"co2":0.5, "cost":0.8}
                }
                prof = profiles[mode]
                st.write(f"- **CO₂/mi:** {prof['co2']} lb")
                st.write(f"- **Cost/mi:** ${prof['cost']}")

            # 6) Route preview & impact
            st.subheader("🔄 Preview & Impact")
            src = geo.query(f"{area_col}==@from_area").iloc[0]
            dst = geo.query(f"{area_col}==@to_area").iloc[0]
            dist = haversine(src["lat"], src["lon"], dst["lat"], dst["lon"])
            co2  = dist * prof["co2"]
            cost = dist * prof["cost"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Distance (mi)", f"{dist:.1f}")
            m2.metric("Est. CO₂ (lb)", f"{co2:.1f}")
            m3.metric("Est. Cost ($)", f"{cost:.2f}")

            # draw line
            import plotly.graph_objects as go
            lf = go.Figure(go.Scattermapbox(
                lat=[src["lat"], dst["lat"]],
                lon=[src["lon"], dst["lon"]],
                mode="lines+markers",
                marker=dict(size=[8,8], color="blue"),
                line=dict(width=3, color="blue"),
            ))
            lf.update_layout(
                mapbox_style="open-street-map",
                mapbox_zoom=4,
                mapbox_center={"lat": (src["lat"]+dst["lat"])/2, "lon": (src["lon"]+dst["lon"])/2},
                margin=dict(l=0,r=0,t=0,b=0)
            )
            st.plotly_chart(lf, use_container_width=True, height=300)

            # 7) Batch Planner
            st.subheader("📋 Batch Transfer Plan")
            if "transfers" not in st.session_state:
                st.session_state["transfers"] = []
            if st.button("➕ Add to Batch"):
                st.session_state["transfers"].append({
                    "From": from_area, "To": to_area, "Qty": qty_move,
                    "Mode": mode, "Dist(mi)": f"{dist:.1f}",
                    "CO₂(lb)": f"{co2:.1f}", "Cost($)": f"{cost:.2f}"
                })
            st.table(pd.DataFrame(st.session_state["transfers"]))

            # 8) Scenario Simulation
            st.subheader("⚡ Scenario Simulation")
            scenario = st.selectbox("Pick a scenario", [
                "Local Festival ➜ spike demand",
                "Snowstorm ➜ route block",
                "Flash Sale ➜ surge orders"
            ])
            if st.button("🔮 Simulate & Advise"):
                batch = pd.DataFrame(st.session_state["transfers"])
                prompt = (
                    f"Scenario: {scenario}.\n"
                    f"Planned transfers:\n{batch.to_string(index=False)}\n\n"
                    "Recommend additional adjustments with rationale (cost/co2)."
                )
                with st.spinner("Asking Gemini…"):
                    advice = get_gemini(prompt)
                st.markdown(advice)

def page_deadstock():
    with st.container():
        st.header("📉 Deadstock Analysis")
        st.markdown(
            "Identify slow-moving inventory items and visualize deadstock across areas."
        )

        if date_col == "–":
            st.warning("Map a **Last Sale Date** column to use deadstock analysis.")
            return

        # 1) Let user define 'dead' threshold
        days_thresh = st.slider(
            "Days since last sale ≥",
            min_value=30, max_value=730, value=180, step=15
        )

        # 2) Filter deadstock
        dead_df = df[df["DaysSinceSale"] >= days_thresh].copy()
        if dead_df.empty:
            st.success(f"No items unsold in the last {days_thresh} days!")
            return

        # 3) Summary metrics
        total_skus   = dead_df[prod_col].nunique()
        total_units  = int(dead_df[qty_col].sum())
        total_value  = dead_df["TotalValue"].sum()
        st.info(f"🔎 Found {total_skus} SKUs, {total_units} units, totaling ${total_value:,.2f}.")

        # 4) Bar chart: deadstock value by area
        st.subheader("Deadstock Value by Area")
        area_dead = (
            dead_df.groupby(area_col)["TotalValue"]
                .sum()
                .reset_index()
                .sort_values("TotalValue", ascending=False)
        )
        fig = px.bar(
            area_dead,
            x=area_col, y="TotalValue",
            labels={"TotalValue":"Value ($)"},
            color="TotalValue",
            color_continuous_scale=px.colors.sequential.Oranges,
            text="TotalValue"
        )
        st.plotly_chart(fig, use_container_width=True, height=350)

        # 5) Detailed table (with search & download)
        st.subheader("Deadstock Items Detail")
        st.dataframe(
            dead_df[[prod_col, qty_col, "TotalValue", "DaysSinceSale", area_col]]
                .sort_values("DaysSinceSale", ascending=False),
            use_container_width=True
        )
        # CSV download
        csv = dead_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Deadstock CSV",
            data=csv,
            file_name="deadstock_items.csv",
            mime="text/csv"
        )

        # 6) Optional AI Insight
        with st.expander("🤖 AI-Powered Deadstock Insights"):
            prompt = (
                f"Here are {total_skus} deadstock SKUs (>= {days_thresh} days since last sale):\n\n"
                f"{dead_df[[prod_col, qty_col, 'TotalValue', 'DaysSinceSale']].head(20).to_string(index=False)}\n\n"
                "Please provide:\n"
                "1. Three key observations about this deadstock.\n"
                "2. Top 3 prioritized actions to reduce deadstock.\n"
                "3. Suggestions to prevent deadstock in the future."
            )
            if st.button("🧠 Generate Deadstock Analysis"):
                with st.spinner("Thinking…"):
                    result = get_gemini(prompt)
                    st.markdown(result)
                        # ── Paste this block after each Gemini response ──
                    feedback_key = f"feedback_{page}"  # unique per page (page can be "deadstock", "restock", "scenario")
                    st.markdown("**Your Feedback**")
                    fb_option = st.radio(
                        "Did you find this helpful?",
                        ["👍 Approve", "👎 Reject"],
                        key=feedback_key + "_radio",
                        horizontal=True
                    )
                    fb_comment = st.text_area(
                        "Any comments or tweaks?",
                        key=feedback_key + "_text",
                        placeholder="Type your feedback here..."
                    )
                    if st.button("Submit Feedback", key=feedback_key+"_btn"):
                        entry = {
                            "page": page,
                            "choice": fb_option,
                            "comment": fb_comment,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.setdefault("feedback_log", []).append(entry)
                        st.success("Thanks for your feedback!")
                        
        st.session_state["deadstock_df"] = dead_df

def page_restock():
    with st.container():
        st.header("💡 Restock & Revive Strategies")
        st.markdown(
            "Turn deadstock into sales with AI-driven action plans and reorder suggestions."
        )

        # Require that deadstock was computed
        dead = st.session_state.get("deadstock_df", None)
        if dead is None or dead.empty:
            st.info("Run **Deadstock Analysis** first to generate a restock plan.")
            return

        # Show summary
        total_skus  = dead[prod_col].nunique()
        total_value = dead["TotalValue"].sum()
        st.write(f"🔢 **Items to address:** {total_skus} SKUs (total value ${total_value:,.2f})")

        # 1) AI-driven Restock Plan
        if st.button("🧠 Generate Restock Action Plan"):
            with st.spinner("Consulting Gemini…"):
                sample = dead[[prod_col, qty_col, "TotalValue", "DaysSinceSale"]].head(20)
                prompt = (
                    f"As an inventory strategist, create a **Restock & Revive** plan "
                    f"for these deadstock SKUs:\n\n{sample.to_string(index=False)}\n\n"
                    f"Total deadstock value: ${total_value:,.2f}\n\n"
                    "Structure your response in Markdown under these headings:\n"
                    "1. Triage & Prioritization (which SKUs first and why)\n"
                    "2. Four actionable marketing/sales strategies (with concrete examples)\n"
                    "3. Operational improvements to avoid future deadstock\n"
                    "4. Final summary recommendation"
                )
                plan = get_gemini(prompt)
                st.markdown(plan)

        # 2) Restock Quantity Recommendations
        st.subheader("Suggested Reorder Quantities")
        st.markdown(
            "Based on average weekly demand, we recommend these reorder levels:"
        )
        # simple heuristic: assume average weekly sales ~ (total units / weeks since added)
        # if no data for weeks since added, default to moving average of last 4 weeks
        weeks_of_data = max(1, int(dead["DaysSinceSale"].max()/7))
        reorder = (
            dead.groupby(prod_col)[qty_col]
                .sum()
                .reset_index(name="TotalUnits")
        )
        reorder["ReorderQty"] = (reorder["TotalUnits"] / weeks_of_data * 2).astype(int).clip(lower=1)
        st.dataframe(reorder[[prod_col, "ReorderQty"]], use_container_width=True)

        # 3) Export Plan
        export_md = plan if "plan" in locals() else ""
        if export_md:
            st.download_button(
                "⬇️ Download Restock Plan (Markdown)",
                data=export_md,
                file_name="restock_plan.md",
                mime="text/markdown"
            )

# --- PAGE ROUTER ---
pages = {
  "Dashboard":       page_dashboard,
  "Area Charts":     page_area_charts,
  "Forecast":        page_forecast,
  "Last-Mile":       page_last_mile,
  "Deadstock":       page_deadstock,
  "Restock":         page_restock,
}
pages[page]()
