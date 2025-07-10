import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(page_title="Energy Density Calculator", layout="wide", page_icon="⚡️")

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stSlider > div > div > div > div {background-color: #4a90e2;}
    .stButton > button {background-color: #4a90e2; color: white; border-radius: 8px;}
    .stButton > button:hover {background-color: #357abd;}
    .header {font-size: 2.5em; color: #2c3e50; text-align: center; margin-bottom: 20px;}
    .subheader {font-size: 1.5em; color: #34495e; margin-top: 20px;}
    .metric-card {background-color: #ffffff; padding: 20px; border-radius: 10px;
                  box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header">Laser Energy Density Calculator</div>', unsafe_allow_html=True)
st.markdown("Calculate **Linear Energy Density (LED)**, **Areal Energy Density (AED)**, and **Volumetric Energy Density (VED)** for laser processing.")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="subheader">Input Parameters</div>', unsafe_allow_html=True)

    # Default state initialization
    if 'laser_power' not in st.session_state:
        st.session_state.laser_power = 250.0
    if 'scan_speed' not in st.session_state:
        st.session_state.scan_speed = 800.0
    if 'hatch_spacing' not in st.session_state:
        st.session_state.hatch_spacing = 70.0
    if 'layer_thickness' not in st.session_state:
        st.session_state.layer_thickness = 30.0

    laser_power = st.slider("Laser Power (Pₗ, W)", 100.0, 500.0, st.session_state.laser_power, step=10.0)
    scan_speed = st.slider("Scan Speed (vₛₐₙ, mm/s)", 500.0, 1500.0, st.session_state.scan_speed, step=10.0)
    hatch_spacing = st.slider("Hatch Spacing (lₕ, μm)", 10.0, 200.0, st.session_state.hatch_spacing, step=5.0)
    layer_thickness = st.slider("Layer Thickness (lₜ, μm)", 10.0, 100.0, st.session_state.layer_thickness, step=5.0)

    # Update state
    st.session_state.laser_power = laser_power
    st.session_state.scan_speed = scan_speed
    st.session_state.hatch_spacing = hatch_spacing
    st.session_state.layer_thickness = layer_thickness

with col2:
    st.markdown('<div class="subheader">Calculated Energy Densities</div>', unsafe_allow_html=True)

    # Calculations
    led = laser_power / scan_speed
    aed = laser_power / (scan_speed * (hatch_spacing / 1000))
    ved = laser_power / (scan_speed * (hatch_spacing / 1000) * (layer_thickness / 1000))

    # Render LaTeX formulas and values
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Linear Energy Density (LED)**")
        st.latex(r"\text{LED} = \frac{P_l}{v_{\text{scan}}}")
        st.write(f"**Value:** {led:.2f} J/mm")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Areal Energy Density (AED)**")
        st.latex(r"\text{AED} = \frac{P_l}{v_{\text{scan}} \cdot l_h}")
        st.write(f"**Value:** {aed:.2f} J/mm²")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Volumetric Energy Density (VED)**")
        st.latex(r"\text{VED} = \frac{P_l}{v_{\text{scan}} \cdot l_h \cdot l_t}")
        st.write(f"**Value:** {ved:.2f} J/mm³")
        st.markdown('</div>', unsafe_allow_html=True)

# Visualization
st.markdown('<div class="subheader">Energy Density Visualization</div>', unsafe_allow_html=True)
st.markdown("Explore how energy density changes with scanning speed and laser power.")

scan_speeds = np.linspace(500, 1500, 100)
laser_powers = [250, 350]

# Precompute data
led_data = {f"{p} W": p / scan_speeds for p in laser_powers}
aed_data = {f"{p} W": p / (scan_speeds * (hatch_spacing / 1000)) for p in laser_powers}
ved_data = {f"{p} W": p / (scan_speeds * (hatch_spacing / 1000) * (layer_thickness / 1000)) for p in laser_powers}

# Plot
fig = go.Figure()
for power in laser_powers:
    fig.add_trace(go.Scatter(x=scan_speeds, y=led_data[f"{power} W"], mode="lines", name=f"LED ({power} W)"))
    fig.add_trace(go.Scatter(x=scan_speeds, y=aed_data[f"{power} W"], mode="lines", name=f"AED ({power} W)"))
    fig.add_trace(go.Scatter(x=scan_speeds, y=ved_data[f"{power} W"], mode="lines", name=f"VED ({power} W)"))

fig.update_layout(
    title="Energy Densities vs. Scanning Speed",
    xaxis_title="Scanning Speed (mm/s)",
    yaxis_title="Energy Density",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Data based on AlSiMg1.4Zr alloy laser processing parameters.")
