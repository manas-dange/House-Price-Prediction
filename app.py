import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="ResiVal | Group 3 House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Theme-Adaptive CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white !important;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px rgba(30, 60, 114, 0.2);
    }
    
    .hero-header h1 {
        margin: 0;
        font-weight: 800;
        font-size: 2.3rem;
        color: white !important;
    }
    
    .hero-header p {
        margin-top: 0.5rem;
        font-size: 1.05rem;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .badge-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        color: white !important;
    }
    
    /* Dynamic Theme-Adaptive Metric Card */
    .metric-card {
        background-color: var(--secondary-background-color, #ffffff) !important;
        color: var(--text-color, #1e293b) !important;
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #3b82f6 !important;
        margin: 0.3rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-color, #64748b) !important;
        opacity: 0.8;
        font-weight: 600;
    }
    
    .confidence-badge {
        font-size: 0.85rem;
        color: #10b981 !important;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15) !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
    }
    
    /* Dynamic Theme-Adaptive Feature Card */
    .feature-card {
        background-color: var(--secondary-background-color, #f8fafc) !important;
        color: var(--text-color, #1e293b) !important;
        border-left: 4px solid #3b82f6 !important;
        border-top: 1px solid rgba(128, 128, 128, 0.25);
        border-right: 1px solid rgba(128, 128, 128, 0.25);
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        padding: 1.2rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .feature-card strong, .feature-card span {
        color: var(--text-color, #1e293b) !important;
    }
    
    /* Dynamic Theme-Adaptive Table */
    .property-table-container {
        background-color: var(--secondary-background-color, #ffffff) !important;
        color: var(--text-color, #1e293b) !important;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .property-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
        color: var(--text-color, #1e293b) !important;
    }
    
    .property-table-container tr {
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    .property-table-container td.label-col {
        padding: 8px;
        color: var(--text-color, #64748b) !important;
        opacity: 0.8;
        font-weight: 600;
    }
    
    .property-table-container td.val-col {
        padding: 8px;
        text-align: right;
        color: var(--text-color, #1e293b) !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_resources():
    model = joblib.load('best_house_price_model.pkl')
    with open('metadata.json', 'r') as f:
        meta = json.load(f)
    return model, meta

try:
    model, metadata = load_resources()
except Exception as e:
    st.error(f"Error loading model resources: {e}. Please ensure you have executed the preceding notebook cells.")
    st.stop()

# Header Banner
st.markdown("""
<div class="hero-header">
    <div style="margin-bottom: 10px;">
        <span class="badge-pill">ML Fundamentals Mini Project</span>
        <span class="badge-pill">Group 3</span>
        <span class="badge-pill">Regression Modeling</span>
    </div>
    <h1>🏡 ResiVal — Intelligent House Valuation Engine</h1>
    <p>Predict residential property selling prices using an optimized Regularized Polynomial Regression model.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: Property Input Parameters
st.sidebar.header("📐 Property Parameters")
st.sidebar.markdown("Configure the house structural & location specifications:")

area_input = st.sidebar.slider(
    "Carpet Area (sq ft)", 
    min_value=600, 
    max_value=4500, 
    value=1850, 
    step=25,
    help="Total interior usable carpet area in square feet."
)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    bedrooms_input = st.number_input("Bedrooms", min_value=1, max_value=6, value=3, step=1)
with col_sb2:
    bathrooms_input = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)

location_input = st.sidebar.selectbox(
    "Neighborhood Location",
    options=metadata["categorical_options"]["location"],
    index=0,
    help="Select the geographic submarket or neighborhood."
)

age_input = st.sidebar.slider(
    "Property Age (Years since construction)",
    min_value=0,
    max_value=45,
    value=8,
    step=1,
    help="Chronological building age impacting physical depreciation."
)

col_sb3, col_sb4 = st.sidebar.columns(2)
with col_sb3:
    parking_input = st.selectbox("Parking Spaces", options=[0, 1, 2, 3], index=1)
with col_sb4:
    stories_input = st.selectbox("Number of Stories", options=[1, 2, 3], index=1)

main_road_input = st.sidebar.radio(
    "Main Road Frontage",
    options=["Yes", "No"],
    index=0,
    horizontal=True,
    help="Direct accessibility to primary municipal arterial roads."
)

furnishing_input = st.sidebar.radio(
    "Furnishing Condition",
    options=metadata["categorical_options"]["furnishing_status"],
    index=1,
    horizontal=True
)

# Predict Button
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("⚡ Estimate Selling Price", type="primary", use_container_width=True)

# Prepare input DataFrame for prediction
input_df = pd.DataFrame([{
    'area': area_input,
    'bedrooms': bedrooms_input,
    'bathrooms': bathrooms_input,
    'location': location_input,
    'age': age_input,
    'parking': parking_input,
    'stories': stories_input,
    'main_road': main_road_input,
    'furnishing_status': furnishing_input
}])

# Compute Prediction
predicted_price = model.predict(input_df)[0]
rmse_val = metadata["metrics"]["test_rmse"]
lower_bound = max(predicted_price - rmse_val, 50000)
upper_bound = predicted_price + rmse_val
price_per_sqft = predicted_price / area_input

# Main Display Area
col_left, col_right = st.columns([1.6, 1.2])

with col_left:
    st.subheader("📊 Valuation Estimate")
    
    # Primary Metrics Card Grid
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Selling Price</div>
            <div class="metric-value">${predicted_price:,.0f}</div>
            <div class="confidence-badge">± ${rmse_val:,.0f} RMSE Margin</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Price per Sq. Ft.</div>
            <div class="metric-value">${price_per_sqft:,.2f}</div>
            <div style="font-size: 0.85rem; color: var(--text-color, #64748b); opacity: 0.8; margin-top: 4px;">Effective Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    # Confidence Interval Alert (Escaped dollar signs for clean markdown parsing)
    st.info(f"📈 **Estimated Valuation Range (68% Confidence)**: **\${lower_bound:,.0f}** — **\${upper_bound:,.0f}** based on our model's holdout RMSE validation.")
    
    # Property Summary Specifications
    st.markdown("### 📋 Property Configuration Snapshot")
    st.markdown(f"""
    <div class="property-table-container">
        <table>
            <tr>
                <td class="label-col">Neighborhood</td>
                <td class="val-col">{location_input}</td>
            </tr>
            <tr>
                <td class="label-col">Living Area</td>
                <td class="val-col">{area_input:,} sq ft</td>
            </tr>
            <tr>
                <td class="label-col">Layout Configuration</td>
                <td class="val-col">{bedrooms_input} Bed | {bathrooms_input} Bath</td>
            </tr>
            <tr>
                <td class="label-col">Building Age</td>
                <td class="val-col">{age_input} years old</td>
            </tr>
            <tr>
                <td class="label-col">Parking & Stories</td>
                <td class="val-col">{parking_input} parking spots | {stories_input} stories</td>
            </tr>
            <tr style="border-bottom: none;">
                <td class="label-col">Road Frontage & Furnishing</td>
                <td class="val-col">Main Road: {main_road_input} | {furnishing_input}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("📍 Market Benchmark")
    
    # Neighborhood Benchmark
    avg_loc_price = metadata["location_averages"].get(location_input, predicted_price)
    diff_pct = ((predicted_price - avg_loc_price) / avg_loc_price) * 100
    diff_sign = "+" if diff_pct >= 0 else ""
    diff_color = "#10b981" if diff_pct >= 0 else "#ef4444"
    
    st.markdown(f"""
    <div class="feature-card">
        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 4px;">Neighborhood Benchmark ({location_input}):</div>
        <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">Average Price: ${avg_loc_price:,.0f}</div>
        <div style="color: {diff_color}; font-weight: 700; font-size: 0.95rem;">
            {diff_sign}{diff_pct:.1f}% vs. neighborhood mean (${avg_loc_price:,.0f})
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Metadata & Academic Info
    with st.expander("🔬 Model Performance Details", expanded=True):
        st.markdown(f"""
        - **Algorithm**: {metadata['model_name']}
        - **Holdout Test R²**: `{metadata['metrics']['test_r2']:.4f}`
        - **Holdout Test RMSE**: `${metadata['metrics']['test_rmse']:,.2f}`
        - **Holdout Test MAE**: `${metadata['metrics']['test_mae']:,.2f}`
        - **Group Attribution**: Group 3 (ML Fundamentals)
        """)
        
    with st.expander("ℹ️ How to Use This Valuation Engine"):
        st.markdown("""
        1. Adjust structural inputs on the left sidebar.
        2. Select neighborhood tier and construction age.
        3. Real-time predictions automatically update!
        4. Cross-check your valuation against the 68% confidence interval.
        """)

st.markdown("---")
st.markdown("<p style='text-align: center; opacity: 0.7; font-size: 0.85rem;'>Developed for Machine Learning Fundamentals (Sem 5) • Group 3 House Price Prediction</p>", unsafe_allow_html=True)