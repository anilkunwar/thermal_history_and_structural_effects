import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(page_title="Energy Density Calculator", layout="wide", page_icon="⚡️")

# Custom CSS styling and MathJax for high-quality LaTeX rendering
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
    .formula {font-size: 1.2em; color: #2c3e50; margin-top: 10px; display: inline;}
    .value {font-size: 1.2em; color: #2c3e50; margin-left: 10px; display: inline;}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true
        }
    });
    </script>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header">Laser Energy Density Calculator</div>', unsafe_allow_html=True)
st.markdown("Calculate **Linear Energy Density (LED)**, **Areal Energy Density (AED)**, and **Volumetric Energy Density (VED)** for laser processing.")
st.markdown("How to use the Dual input interface conveniently: Change in **input box** automatically updates the slider, The change in **slider** has to be followed by an attempt to retreat and the input in box gets updated.")

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
        label="Scanning Speed ($v_{\\text{scan}}$ mm/s)",
        min_value=500.0,
        max_value=1500.0,
        value=st.session_state.scan_speed,
        step=10.0,
        key="scan_speed_input",
        help="Scanning speed in mm/s (default: 800 mm/s)"
    )
    st.session_state.scan_speed = scan_speed_input
    scan_speed = st.slider(
        label="Scanning Speed ($v_{\\text{scan}}$ mm/s)",
        min_value=500.0,
        max_value=1500.0,
        value=st.session_state.scan_speed,
        step=10.0,
        key="scan_speed_slider",
        help="Adjust scanning speed"
    )
    st.session_state.scan_speed = scan_speed

    hatch_spacing_input = st.number_input(
        "Hatch Spacing ($l_{\\text{h}}$, μm)",
        min_value=10.0,
        max_value=200.0,
        value=st.session_state.hatch_spacing,
        step=5.0,
        key="hatch_spacing_input",
        help="Hatch spacing in micrometers (default: 70 μm)"
    )
    st.session_state.hatch_spacing = hatch_spacing_input
    hatch_spacing = st.slider(
        "Hatch Spacing Slider ($l_{\\text{h}}$, μm)",
        min_value=10.0,
        max_value=200.0,
        value=st.session_state.hatch_spacing,
        step=5.0,
        key="hatch_spacing_slider",
        help="Adjust hatch spacing"
    )
    st.session_state.hatch_spacing = hatch_spacing

    layer_thickness_input = st.number_input(
        "Layer Thickness (($l_{\\text{t}}$, μm))",
        min_value=10.0,
        max_value=100.0,
        value=st.session_state.layer_thickness,
        step=5.0,
        key="layer_thickness_input",
        help="Layer thickness in micrometers (default: 30 μm)"
    )
    st.session_state.layer_thickness = layer_thickness_input
    layer_thickness = st.slider(
        "Layer Thickness Slider (($l_{\\text{t}}$, μm))",
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

    # Calculations
    led = laser_power / scan_speed
    aed = laser_power / (scan_speed * (hatch_spacing / 1000))
    ved = laser_power / (scan_speed * (hatch_spacing / 1000) * (layer_thickness / 1000))

    # Render LaTeX formulas with numerical values and final results
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Linear Energy Density (LED)**")
        st.markdown(
            f'<span class="formula">$ \\text{{LED}} = \\frac{{P_l}}{{v_{{\\text{{scan}}}}}} = \\frac{{{laser_power:.1f}}}{{{scan_speed:.1f}}} \\approx {led:.2f} \\, \\text{{J/mm}}$</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Final Value:** {led:.2f} J/mm")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Areal Energy Density (AED)**")
        st.markdown(
            f'<span class="formula">$ \\text{{AED}} = \\frac{{P_l}}{{v_{{\\text{{scan}}}} \\cdot l_h}} = \\frac{{{laser_power:.1f}}}{{{scan_speed:.1f} \\cdot {hatch_spacing/1000:.4f}}} \\approx {aed:.2f} \\, \\text{{J/mm}}^2$</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Final Value:** {aed:.2f} J/mm²")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Volumetric Energy Density (VED)**")
        st.markdown(
            f'<span class="formula">$ \\text{{VED}} = \\frac{{P_l}}{{v_{{\\text{{scan}}}} \\cdot l_h \\cdot l_t}} = \\frac{{{laser_power:.1f}}}{{{scan_speed:.1f} \\cdot {hatch_spacing/1000:.4f} \\cdot {layer_thickness/1000:.4f}}} \\approx {ved:.2f} \\, \\text{{J/mm}}^3$</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Final Value:** {ved:.2f} J/mm³")
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
