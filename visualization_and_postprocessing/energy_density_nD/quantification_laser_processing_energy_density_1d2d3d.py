import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Laser Energy Density Calculator",
    layout="wide",
    page_icon="⚡️",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    .stSlider > div > div > div > div {background-color: #3b82f6;}
    .stButton > button {background-color: #3b82f6; color: white; border-radius: 8px; font-weight: 500;}
    .stButton > button:hover {background-color: #2563eb;}
    .header {font-size: 2.5em; color: #1e293b; text-align: center; margin-bottom: 10px; font-weight: 700;}
    .subheader {font-size: 1.4em; color: #334155; margin-top: 25px; margin-bottom: 15px; font-weight: 600; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;}
    .info-text {color: #64748b; font-size: 0.9em; margin-bottom: 10px;}
    .footer {text-align: center; color: #94a3b8; font-size: 0.85em; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CALLBACKS FOR DUAL INPUT SYNC
# ============================================================
def sync_num_to_slider(key):
    st.session_state[f"{key}_slider"] = st.session_state[f"{key}_num"]

def sync_slider_to_num(key):
    st.session_state[f"{key}_num"] = st.session_state[f"{key}_slider"]

# Initialize session state
for key, val in {
    "laser_power_num": 250.0, "laser_power_slider": 250.0,
    "scan_speed_num": 800.0, "scan_speed_slider": 800.0,
    "hatch_spacing_num": 70.0, "hatch_spacing_slider": 70.0,
    "layer_thickness_num": 30.0, "layer_thickness_slider": 30.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# SCHEMATIC GENERATOR
# ============================================================
def create_schematic(power, speed, hatch, thickness):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top View: Scan Strategy", "Side View: Layer Geometry"),
        horizontal_spacing=0.08
    )
    
    # --- TOP VIEW (col=1) ---
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=8, row=1, col=1,
                  fillcolor="#f1f5f9", line=dict(color="#94a3b8", width=3), layer="below")
    
    n_lines = 5
    y_pos = np.linspace(1.5, 6.5, n_lines)
    for i, y in enumerate(y_pos):
        # Scan track
        fig.add_shape(type="line", x0=1, y0=y, x1=9, y1=y, row=1, col=1,
                      line=dict(color="#3b82f6", width=3))
        # Scan-direction arrows (bi-directional)
        if i % 2 == 0:
            fig.add_annotation(x=8.5, y=y+0.25, ax=1.5, ay=y+0.25, row=1, col=1,
                               showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#1e40af")
        else:
            fig.add_annotation(x=1.5, y=y+0.25, ax=8.5, ay=y+0.25, row=1, col=1,
                               showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#1e40af")
    
    # Hatch spacing bracket
    fig.add_shape(type="line", x0=9.8, y0=y_pos[0], x1=9.8, y1=y_pos[1], row=1, col=1,
                  line=dict(color="#10b981", width=2))
    fig.add_shape(type="line", x0=9.6, y0=y_pos[0], x1=10, y1=y_pos[0], row=1, col=1,
                  line=dict(color="#10b981", width=2))
    fig.add_shape(type="line", x0=9.6, y0=y_pos[1], x1=10, y1=y_pos[1], row=1, col=1,
                  line=dict(color="#10b981", width=2))
    fig.add_annotation(x=10.4, y=(y_pos[0]+y_pos[1])/2, text=f"l<sub>h</sub>={hatch}μm",
                       row=1, col=1, showarrow=False, font=dict(color="#059669", size=11))
    
    # Laser spot on middle track
    fig.add_shape(type="circle", x0=4.2, y0=y_pos[2]-0.4, x1=5.8, y1=y_pos[2]+0.4, row=1, col=1,
                  fillcolor="rgba(239,68,68,0.25)", line=dict(color="#dc2626", width=2))
    fig.add_annotation(x=5, y=y_pos[2], text=f"P={power}W", row=1, col=1,
                       showarrow=False, font=dict(size=10, color="#991b1b"))
    fig.add_annotation(x=5, y=7.8, text=f"v<sub>scan</sub>={speed} mm/s", row=1, col=1,
                       showarrow=False, font=dict(color="#1e40af", size=12))
    
    # --- SIDE VIEW (col=2) ---
    # Substrate / prior layers
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=3, row=1, col=2,
                  fillcolor="#64748b", line=dict(color="#475569", width=2), layer="below")
    fig.add_annotation(x=5, y=1.5, text="Substrate / Prior Layers", row=1, col=2,
                       showarrow=False, font=dict(color="white", size=11))
    
    # Current powder layer (thickness exaggerated for visibility)
    t_vis = max(0.4, thickness / 25)
    fig.add_shape(type="rect", x0=0, y0=3, x1=10, y1=3+t_vis, row=1, col=2,
                  fillcolor="#fbbf24", line=dict(color="#d97706", width=2), opacity=0.85, layer="below")
    fig.add_annotation(x=5, y=3+t_vis/2, text="Current Layer", row=1, col=2,
                       showarrow=False, font=dict(color="#92400e", size=11))
    
    # Laser beam cone
    beam_top = 3 + t_vis + 3.5
    fig.add_shape(type="path", path=f"M 3.5,{beam_top} L 5,{3+t_vis} L 6.5,{beam_top} Z", row=1, col=2,
                  fillcolor="rgba(239,68,68,0.12)", line=dict(color="#dc2626", width=1, dash="dot"))
    fig.add_shape(type="ellipse", x0=4.5, y0=3+t_vis-0.12, x1=5.5, y1=3+t_vis+0.12, row=1, col=2,
                  fillcolor="rgba(239,68,68,0.35)", line=dict(color="#dc2626", width=2))
    
    # Layer thickness dimension
    fig.add_shape(type="line", x0=-0.5, y0=3, x1=-0.5, y1=3+t_vis, row=1, col=2,
                  line=dict(color="#7c3aed", width=2))
    fig.add_shape(type="line", x0=-0.7, y0=3, x1=-0.3, y1=3, row=1, col=2,
                  line=dict(color="#7c3aed", width=2))
    fig.add_shape(type="line", x0=-0.7, y0=3+t_vis, x1=-0.3, y1=3+t_vis, row=1, col=2,
                  line=dict(color="#7c3aed", width=2))
    fig.add_annotation(x=-1.1, y=3+t_vis/2, text=f"l<sub>t</sub>={thickness}μm", row=1, col=2,
                       showarrow=False, font=dict(color="#6d28d9", size=11), textangle=-90)
    fig.add_annotation(x=5, y=beam_top+0.3, text=f"P<sub>l</sub>={power} W", row=1, col=2,
                       showarrow=False, font=dict(color="#991b1b", size=12))
    
    fig.update_layout(
        height=400, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif")
    )
    for c in [1, 2]:
        fig.update_xaxes(visible=False, range=[-1.5, 11.5], row=1, col=c)
        fig.update_yaxes(visible=False, range=[-0.5, 9], row=1, col=c, scaleanchor="x", scaleratio=1)
    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="header">⚡ Laser Energy Density Calculator</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #475569; margin-bottom: 25px; max-width: 800px; margin-left: auto; margin-right: auto;">
    Calculate <b>Linear (LED)</b>, <b>Areal (AED)</b>, and <b>Volumetric (VED)</b> 
    Energy Density for Laser Powder Bed Fusion (LPBF) and directed energy deposition processes.
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN COLUMNS
# ============================================================
col_inputs, col_results = st.columns([1, 1.2], gap="large")

with col_inputs:
    st.markdown('<div class="subheader">🔧 Process Parameters</div>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Number inputs and sliders are synchronized. Edit either one.</p>', unsafe_allow_html=True)
    
    # --- Laser Power ---
    c1, c2 = st.columns([1, 2])
    with c1:
        st.number_input("Power (W)", min_value=100.0, max_value=1000.0, step=10.0,
                        key="laser_power_num", on_change=sync_num_to_slider, args=("laser_power",),
                        help="Laser power in watts")
    with c2:
        st.slider("Laser Power", min_value=100.0, max_value=1000.0, step=10.0,
                  key="laser_power_slider", on_change=sync_slider_to_num, args=("laser_power",),
                  help="Adjust laser power")
    laser_power = st.session_state.laser_power_slider
    st.latex(r"P_l = " + f"{laser_power:.1f}" + r"~\text{W}")
    
    # --- Scan Speed ---
    c1, c2 = st.columns([1, 2])
    with c1:
        st.number_input("Speed (mm/s)", min_value=100.0, max_value=1500.0, step=10.0,
                        key="scan_speed_num", on_change=sync_num_to_slider, args=("scan_speed",),
                        help="Scanning speed in mm/s")
    with c2:
        st.slider("Scan Speed", min_value=100.0, max_value=1500.0, step=10.0,
                  key="scan_speed_slider", on_change=sync_slider_to_num, args=("scan_speed",),
                  help="Adjust scanning speed")
    scan_speed = st.session_state.scan_speed_slider
    st.latex(r"v_{\text{scan}} = " + f"{scan_speed:.1f}" + r"~\text{mm/s}")
    
    # --- Hatch Spacing ---
    c1, c2 = st.columns([1, 2])
    with c1:
        st.number_input("Hatch (μm)", min_value=10.0, max_value=200.0, step=5.0,
                        key="hatch_spacing_num", on_change=sync_num_to_slider, args=("hatch_spacing",),
                        help="Hatch spacing in micrometers")
    with c2:
        st.slider("Hatch Spacing", min_value=10.0, max_value=200.0, step=5.0,
                  key="hatch_spacing_slider", on_change=sync_slider_to_num, args=("hatch_spacing",),
                  help="Distance between adjacent scan tracks")
    hatch_spacing = st.session_state.hatch_spacing_slider
    st.latex(r"l_h = " + f"{hatch_spacing:.1f}" + r"~\mu\text{m} = " + f"{hatch_spacing/1000:.4f}" + r"~\text{mm}")
    
    # --- Layer Thickness ---
    c1, c2 = st.columns([1, 2])
    with c1:
        st.number_input("Thickness (μm)", min_value=10.0, max_value=100.0, step=5.0,
                        key="layer_thickness_num", on_change=sync_num_to_slider, args=("layer_thickness",),
                        help="Layer thickness in micrometers")
    with c2:
        st.slider("Layer Thickness", min_value=10.0, max_value=100.0, step=5.0,
                  key="layer_thickness_slider", on_change=sync_slider_to_num, args=("layer_thickness",),
                  help="Thickness of each deposited layer")
    layer_thickness = st.session_state.layer_thickness_slider
    st.latex(r"l_t = " + f"{layer_thickness:.1f}" + r"~\mu\text{m} = " + f"{layer_thickness/1000:.4f}" + r"~\text{mm}")
    
    # --- Process Schematic ---
    st.markdown('<div class="subheader">📐 Process Schematic</div>', unsafe_allow_html=True)
    st.plotly_chart(create_schematic(laser_power, scan_speed, hatch_spacing, layer_thickness), use_container_width=True)

with col_results:
    st.markdown('<div class="subheader">📈 Energy Density Results</div>', unsafe_allow_html=True)
    
    # Unit conversions
    h_mm = hatch_spacing / 1000.0
    t_mm = layer_thickness / 1000.0
    
    # Calculations
    led = laser_power / scan_speed
    aed = laser_power / (scan_speed * h_mm)
    ved = laser_power / (scan_speed * h_mm * t_mm)
    
    st.markdown("**Current Parameter Summary**")
    st.latex(rf"P_l = {laser_power:.1f}~\text{{W}}, \quad v_{{\text{{scan}}}} = {scan_speed:.1f}~\text{{mm/s}}, \quad l_h = {hatch_spacing:.1f}~\mu\text{{m}}, \quad l_t = {layer_thickness:.1f}~\mu\text{{m}}")
    st.divider()
    
    # --- LED Card ---
    with st.container(border=True):
        st.markdown("#### 🔵 Linear Energy Density (LED)")
        st.caption("Energy deposited per unit length along a single scan track")
        st.latex(r"\text{LED} = \frac{P_l}{v_{\text{scan}}} \quad [\text{J/mm}]")
        st.latex(r"\text{LED} = \frac{" + f"{laser_power:.1f}" + r"~\text{W}}{" + f"{scan_speed:.1f}" + r"~\text{mm/s}} \approx " + f"{led:.3f}" + r"~\text{J/mm}")
        st.metric("Computed LED", f"{led:.3f} J/mm")
    
    # --- AED Card ---
    with st.container(border=True):
        st.markdown("#### 🔴 Areal Energy Density (AED)")
        st.caption("Energy deposited per unit area of the scan plane (track × hatch)")
        st.latex(r"\text{AED} = \frac{P_l}{v_{\text{scan}} \cdot l_h} \quad [\text{J/mm}^2]")
        st.latex(r"\text{AED} = \frac{" + f"{laser_power:.1f}" + r"}{" + f"{scan_speed:.1f} \times {h_mm:.4f}" + r"} \approx " + f"{aed:.3f}" + r"~\text{J/mm}^2")
        st.metric("Computed AED", f"{aed:.3f} J/mm²")
    
    # --- VED Card ---
    with st.container(border=True):
        st.markdown("#### 🟢 Volumetric Energy Density (VED)")
        st.caption("Energy deposited per unit volume of the deposited layer")
        st.latex(r"\text{VED} = \frac{P_l}{v_{\text{scan}} \cdot l_h \cdot l_t} \quad [\text{J/mm}^3]")
        st.latex(r"\text{VED} = \frac{" + f"{laser_power:.1f}" + r"}{" + f"{scan_speed:.1f} \times {h_mm:.4f} \times {t_mm:.4f}" + r"} \approx " + f"{ved:.3f}" + r"~\text{J/mm}^3")
        st.metric("Computed VED", f"{ved:.3f} J/mm³")

# ============================================================
# VISUALIZATION SECTION
# ============================================================
st.markdown('<div class="subheader">📊 Energy Density Visualization</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📉 Line Chart: Speed Sweep", "🗺️ Contour Map: Process Window"])

with tab1:
    st.markdown("Explore how energy densities vary with scanning speed for different laser powers. Your current speed is marked.")
    scan_speeds = np.linspace(100, 1500, 150)
    powers = [150, 250, 350, 450]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    
    fig1 = go.Figure()
    for p, color in zip(powers, colors):
        led_line = p / scan_speeds
        aed_line = p / (scan_speeds * h_mm)
        ved_line = p / (scan_speeds * h_mm * t_mm)
        
        fig1.add_trace(go.Scatter(x=scan_speeds, y=led_line, mode="lines", name=f"LED ({p} W)",
                                    line=dict(color=color, width=2, dash="solid"), opacity=0.9))
        fig1.add_trace(go.Scatter(x=scan_speeds, y=aed_line, mode="lines", name=f"AED ({p} W)",
                                    line=dict(color=color, width=2, dash="dash"), opacity=0.9))
        fig1.add_trace(go.Scatter(x=scan_speeds, y=ved_line, mode="lines", name=f"VED ({p} W)",
                                    line=dict(color=color, width=2, dash="dot"), opacity=0.9))
    
    fig1.add_vline(x=scan_speed, line=dict(color="black", width=1, dash="dot"),
                   annotation_text="Current speed", annotation_position="top")
    fig1.update_layout(
        title="Energy Densities vs. Scanning Speed",
        xaxis_title="Scanning Speed (mm/s)",
        yaxis_title="Energy Density",
        template="plotly_white",
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.markdown("Interactive process map showing VED as a function of laser power and scan speed. The white **×** marks your current settings.")
    power_range = np.linspace(100, 500, 80)
    speed_range = np.linspace(100, 1500, 80)
    P_grid, V_grid = np.meshgrid(power_range, speed_range)
    VED_grid = P_grid / (V_grid * h_mm * t_mm)
    
    fig2 = go.Figure(data=go.Contour(
        z=VED_grid, x=power_range, y=speed_range,
        colorscale="Plasma", ncontours=20,
        contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=10, color="white")),
        colorbar=dict(title="VED (J/mm³)", titleside="right"),
        hovertemplate="Power: %{x:.0f} W<br>Speed: %{y:.0f} mm/s<br>VED: %{z:.2f} J/mm³<<extra></extra>"
    ))
    fig2.add_trace(go.Scatter(
        x=[laser_power], y=[scan_speed], mode="markers+text",
        marker=dict(color="white", size=14, symbol="x", line=dict(color="black", width=2)),
        text=["Current"], textposition="top center", name="Current Setting"
    ))
    fig2.update_layout(
        xaxis_title="Laser Power (W)", yaxis_title="Scan Speed (mm/s)",
        height=550, template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# REFERENCE & FOOTER
# ============================================================
with st.expander("📖 Reference Values & Physical Interpretation"):
    col_ref1, col_ref2, col_ref3 = st.columns(3)
    with col_ref1:
        st.markdown("**Linear Energy Density (LED)**")
        st.latex(r"\text{LED} = \frac{P_l}{v_{\text{scan}}}")
        st.markdown("Energy per unit length along one scan track. Typical LPBF values: **0.2 – 0.6 J/mm**.")
    with col_ref2:
        st.markdown("**Areal Energy Density (AED)**")
        st.latex(r"\text{AED} = \frac{P_l}{v_{\text{scan}} \cdot l_h}")
        st.markdown("Energy per unit area of the scan plane. Typical LPBF values: **3 – 12 J/mm²**.")
    with col_ref3:
        st.markdown("**Volumetric Energy Density (VED)**")
        st.latex(r"\text{VED} = \frac{P_l}{v_{\text{scan}} \cdot l_h \cdot l_t}")
        st.markdown("Energy per unit volume of the build layer. Typical LPBF values: **50 – 200 J/mm³**.")

st.markdown("---")
st.markdown("""
<div class="footer">
    <b>Data context:</b> AlSiMg1.4Zr alloy laser processing parameters.<br>
    <b>Citation:</b> X. Wang, Y. Geng, Y. Oliinyk, Z. Zhang, A. Kunwar, 
    <i>Multiscale Computation and Experimental Insights into Thermal History and Composition based Study of Strength-Ductility Synergy in Zr-Enhanced AlSiMg Alloys</i>, 2025.
</div>
""", unsafe_allow_html=True)
