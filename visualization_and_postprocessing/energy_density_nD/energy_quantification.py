import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Set page configuration for a modern look
st.set_page_config(page_title="Energy Density Calculator", layout="wide", page_icon="⚡️")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stSlider > div > div > div > div {background-color: #4a90e2;}
    .stButton > button {background-color: #4a90e2; color: white; border-radius: 8px;}
    .stButton > button:hover {background-color: #357abd;}
    .metric-card {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 15px;}
    .header {font-size: 2.5em; color: #2c3e50; text-align: center; margin-bottom: 20px;}
    .subheader {font-size: 1.5em; color: #34495e; margin-top: 20px;}
    .formula {font-size: 1.2em; color: #2c3e50; margin-top: 10px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header">Laser Energy Density Calculator</div>', unsafe_allow_html=True)
st.markdown("Calculate **Linear Energy Density (LED)**, **Areal Energy Density (AED)**, and **Volumetric Energy Density (VED)** for laser processing.")

# Layout with columns for inputs and results
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="subheader">Input Parameters</div>', unsafe_allow_html=True)
    
    # Initialize session state for inputs
    if 'laser_power' not in st.session_state:
        st.session_state.laser_power = 250.0
    if 'scan_speed' not in st.session_state:
        st.session_state.scan_speed = 800.0
    if 'hatch_spacing' not in st.session_state:
        st.session_state.hatch_spacing = 70.0
    if 'layer_thickness' not in st.session_state:
        st.session_state.layer_thickness = 30.0

    # Input boxes and sliders
    laser_power_input = st.number_input(
        "Laser Power (P_l, W)",
        min_value=100.0,
        max_value=500.0,
        value=st.session_state.laser_power,
        step=10.0,
        key="laser_power_input",
        help="Laser power in watts (default: 250 W)"
    )
    st.session_state.laser_power = laser_power_input
    laser_power = st.slider(
        "Laser Power Slider (P_l, W)",
        min_value=100.0,
        max_value=500.0,
        value=st.session_state.laser_power,
        step=10.0,
        key="laser_power_slider",
        help="Adjust laser power"
    )
    st.session_state.laser_power = laser_power

    scan_speed_input = st.number_input(
        "Scanning Speed (v_scan, mm/s)",
        min_value=500.0,
        max_value=1500.0,
        value=st.session_state.scan_speed,
        step=10.0,
        key="scan_speed_input",
        help="Scanning speed in mm/s (default: 800 mm/s)"
    )
    st.session_state.scan_speed = scan_speed_input
    scan_speed = st.slider(
        "Scanning Speed Slider (v_scan, mm/s)",
        min_value=500.0,
        max_value=1500.0,
        value=st.session_state.scan_speed,
        step=10.0,
        key="scan_speed_slider",
        help="Adjust scanning speed"
    )
    st.session_state.scan_speed = scan_speed

    hatch_spacing_input = st.number_input(
        "Hatch Spacing (l_h, μm)",
        min_value=10.0,
        max_value=200.0,
        value=st.session_state.hatch_spacing,
        step=5.0,
        key="hatch_spacing_input",
        help="Hatch spacing in micrometers (default: 70 μm)"
    )
    st.session_state.hatch_spacing = hatch_spacing_input
    hatch_spacing = st.slider(
        "Hatch Spacing Slider (l_h, μm)",
        min_value=10.0,
        max_value=200.0,
        value=st.session_state.hatch_spacing,
        step=5.0,
        key="hatch_spacing_slider",
        help="Adjust hatch spacing"
    )
    st.session_state.hatch_spacing = hatch_spacing

    layer_thickness_input = st.number_input(
        "Layer Thickness (l_t, μm)",
        min_value=10.0,
        max_value=100.0,
        value=st.session_state.layer_thickness,
        step=5.0,
        key="layer_thickness_input",
        help="Layer thickness in micrometers (default: 30 μm)"
    )
    st.session_state.layer_thickness = layer_thickness_input
    layer_thickness = st.slider(
        "Layer Thickness Slider (l_t, μm)",
        min_value=10.0,
        max_value=100.0,
        value=st.session_state.layer_thickness,
        step=5.0,
        key="layer_thickness_slider",
        help="Adjust layer thickness"
    )
    st.session_state.layer_thickness = layer_thickness

with col2:
    st.markdown('<div class="subheader">Calculated Energy Densities</div>', unsafe_allow_html=True)
    
    # Calculate energy density values
    led = laser_power / scan_speed  # LED = P_l / v_scan (J/mm)
    aed = laser_power / (scan_speed * (hatch_spacing / 1000))  # AED = P_l / (v_scan * l_h) (J/mm²)
    ved = laser_power / (scan_speed * (hatch_spacing / 1000) * (layer_thickness / 1000))  # VED = P_l / (v_scan * l_h * l_t) (J/mm³)
    
    # Display formulas and results in styled metric cards
    st.markdown(
        f"""
        <div class="metric-card">
            <b>Linear Energy Density (LED):</b><br>
            <div class="formula">Formula: $$ \\text{{LED}} = \\frac{{P_l}}{{v_{{scan}}}} $$</div>
            <b>Value:</b> {led:.2f} J/mm
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <b>Areal Energy Density (AED):</b><br>
            <div class="formula">Formula: $$ \\text{{AED}} = \\frac{{P_l}}{{v_{{scan}} \\times l_h}} $$</div>
            <b>Value:</b> {aed:.2f} J/mm²
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <b>Volumetric Energy Density (VED):</b><br>
            <div class="formula">Formula: $$ \\text{{VED}} = \\frac{{P_l}}{{v_{{scan}} \\times l_h \\times l_t}} $$</div>
            <b>Value:</b> {ved:.2f} J/mm³
        </div>
        """,
        unsafe_allow_html=True
    )

# Plotting section
st.markdown('<div class="subheader">Energy Density Visualization</div>', unsafe_allow_html=True)
st.markdown("Explore how energy density changes with scanning speed and laser power.")

# Create a range of scanning speeds and laser powers for plotting
scan_speeds = np.linspace(500, 1500, 100)
laser_powers = [250, 350]  # Default powers from the text
led_data = {f"{p} W": p / scan_speeds for p in laser_powers}
aed_data = {f"{p} W": p / (scan_speeds * (hatch_spacing / 1000)) for p in laser_powers}
ved_data = {f"{p} W": p / (scan_speeds * (hatch_spacing / 1000) * (layer_thickness / 1000)) for p in laser_powers}

# Create Plotly figure
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
