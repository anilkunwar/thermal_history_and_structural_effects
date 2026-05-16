import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json

# ============================================================
# PAGE CONFIG
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
    .subheader {font-size: 1.4em; color: #334155; margin-top: 25px; margin-bottom: 15px; font-weight: 600;
                border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;}
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
_defaults = {
    "laser_power_num": 250.0, "laser_power_slider": 250.0,
    "scan_speed_num": 800.0, "scan_speed_slider": 800.0,
    "hatch_spacing_num": 70.0, "hatch_spacing_slider": 70.0,
    "layer_thickness_num": 30.0, "layer_thickness_slider": 30.0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SCHEMATIC (100% go.Scatter — no add_shape quirks)
# ============================================================
def create_schematic(P, v, h_um, t_um):
    h_mm = h_um / 1000.0
    t_mm = t_um / 1000.0

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📐 Top View: Scan Strategy", "🔍 Side View: Layer Geometry"),
        horizontal_spacing=0.12
    )

    # ---------- TOP VIEW (col 1) ----------
    # Build plate
    fig.add_trace(go.Scatter(
        x=[0, 10, 10, 0, 0], y=[0, 0, 8, 8, 0],
        mode="lines", fill="toself", fillcolor="rgba(241,245,249,0.6)",
        line=dict(color="#94a3b8", width=3),
        name="Build Plate", hoverinfo="skip"
    ), row=1, col=1)

    # Scan tracks
    n_lines = 5
    y_tracks = np.linspace(1.6, 6.4, n_lines)
    for i, y in enumerate(y_tracks):
        fig.add_trace(go.Scatter(
            x=[1, 9], y=[y, y], mode="lines",
            line=dict(color="#3b82f6", width=2.5),
            name="Scan Track" if i == 0 else None,
            showlegend=(i == 0), hoverinfo="skip"
        ), row=1, col=1)

        # Scan-direction arrows (bi-directional)
        if i % 2 == 0:
            fig.add_annotation(
                x=8.5, y=y + 0.18, ax=1.5, ay=y + 0.18,
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
                arrowcolor="#1e40af", row=1, col=1
            )
        else:
            fig.add_annotation(
                x=1.5, y=y + 0.18, ax=8.5, ay=y + 0.18,
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
                arrowcolor="#1e40af", row=1, col=1
            )

    # Hatch spacing bracket (between first two tracks)
    y1, y2 = y_tracks[0], y_tracks[1]
    fig.add_trace(go.Scatter(x=[9.5, 9.5], y=[y1, y2], mode="lines",
                               line=dict(color="#10b981", width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[9.3, 9.7], y=[y1, y1], mode="lines",
                               line=dict(color="#10b981", width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[9.3, 9.7], y=[y2, y2], mode="lines",
                               line=dict(color="#10b981", width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_annotation(x=10.1, y=(y1 + y2) / 2, text=f"<b>l<sub>h</sub></b>={h_um:.0f} μm",
                       showarrow=False, font=dict(color="#059669", size=11),
                       row=1, col=1)

    # Laser spot (parametric circle via scatter fill)
    theta = np.linspace(0, 2 * np.pi, 60)
    cx, cy, r = 5.0, y_tracks[2], 0.55
    fig.add_trace(go.Scatter(
        x=cx + r * np.cos(theta), y=cy + r * np.sin(theta),
        mode="lines", fill="toself", fillcolor="rgba(239,68,68,0.25)",
        line=dict(color="#dc2626", width=2),
        name="Laser Spot", hoverinfo="skip"
    ), row=1, col=1)
    fig.add_annotation(x=cx, y=cy, text=f"<b>{P:.0f} W</b>",
                       showarrow=False, font=dict(size=10, color="#991b1b"),
                       row=1, col=1)
    fig.add_annotation(x=5, y=7.5, text=f"<b>v<sub>scan</sub></b> = {v:.0f} mm/s",
                       showarrow=False, font=dict(color="#1e40af", size=12),
                       row=1, col=1)

    # ---------- SIDE VIEW (col 2) ----------
    t_vis = max(0.6, t_mm * 25)  # exaggerated for visibility

    # Substrate block
    fig.add_trace(go.Scatter(
        x=[0, 10, 10, 0, 0], y=[0, 0, 3, 3, 0],
        mode="lines", fill="toself", fillcolor="#64748b",
        line=dict(color="#475569", width=2),
        name="Substrate", hoverinfo="skip"
    ), row=1, col=2)
    fig.add_annotation(x=5, y=1.5, text="Substrate / Prior Layers",
                       showarrow=False, font=dict(color="white", size=12),
                       row=1, col=2)

    # Current layer block
    fig.add_trace(go.Scatter(
        x=[0, 10, 10, 0, 0], y=[3, 3, 3 + t_vis, 3 + t_vis, 3],
        mode="lines", fill="toself", fillcolor="rgba(251,191,36,0.85)",
        line=dict(color="#d97706", width=2),
        name="Powder Layer", hoverinfo="skip"
    ), row=1, col=2)
    fig.add_annotation(x=5, y=3 + t_vis / 2, text="Current Layer",
                       showarrow=False, font=dict(color="#92400e", size=12),
                       row=1, col=2)

    # Laser beam cone
    beam_h = 3.5
    fig.add_trace(go.Scatter(
        x=[4.2, 5.0, 5.8, 4.2],
        y=[3 + t_vis + beam_h, 3 + t_vis, 3 + t_vis + beam_h, 3 + t_vis + beam_h],
        mode="lines", fill="toself", fillcolor="rgba(239,68,68,0.10)",
        line=dict(color="#dc2626", width=1, dash="dot"),
        name="Laser Beam", hoverinfo="skip"
    ), row=1, col=2)

    # Laser spot on surface (flattened ellipse via scatter)
    r2 = 0.25
    fig.add_trace(go.Scatter(
        x=5.0 + r2 * np.cos(theta),
        y=3 + t_vis + 0.25 * r2 * np.sin(theta),
        mode="lines", fill="toself", fillcolor="rgba(239,68,68,0.4)",
        line=dict(color="#dc2626", width=2),
        showlegend=False, hoverinfo="skip"
    ), row=1, col=2)

    # Layer thickness dimension lines
    fig.add_trace(go.Scatter(x=[-0.6, -0.6], y=[3, 3 + t_vis], mode="lines",
                               line=dict(color="#7c3aed", width=2), hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[-0.8, -0.4], y=[3, 3], mode="lines",
                               line=dict(color="#7c3aed", width=2), hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[-0.8, -0.4], y=[3 + t_vis, 3 + t_vis], mode="lines",
                               line=dict(color="#7c3aed", width=2), hoverinfo="skip"), row=1, col=2)
    fig.add_annotation(x=-1.3, y=3 + t_vis / 2, text=f"<b>l<sub>t</sub></b>={t_um:.0f} μm",
                       showarrow=False, font=dict(color="#6d28d9", size=11),
                       textangle=-90, row=1, col=2)

    fig.add_annotation(x=5, y=3 + t_vis + beam_h + 0.4,
                       text=f"<b>P<sub>l</sub></b> = {P:.0f} W",
                       showarrow=False, font=dict(color="#991b1b", size=12),
                       row=1, col=2)

    # Global layout
    fig.update_layout(
        height=430, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif")
    )

    for c in [1, 2]:
        fig.update_xaxes(visible=False, range=[-2, 12], row=1, col=c)
        fig.update_yaxes(visible=False, range=[-1, 10], row=1, col=c,
                         scaleanchor="x", scaleratio=1)

    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="header">⚡ Laser Energy Density Calculator</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #475569; margin-bottom: 25px; max-width: 800px; margin-left: auto; margin-right: auto;">
    Calculate <b>Linear (LED)</b>, <b>Areal (AED)</b>, and <b>Volumetric (VED)</b> 
    Energy Density for Laser Powder Bed Fusion (LPBF) and additive manufacturing processes.
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN COLUMNS: INPUTS | RESULTS
# ============================================================
col_inputs, col_results = st.columns([1, 1.15], gap="large")

with col_inputs:
    st.markdown('<div class="subheader">🔧 Process Parameters</div>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Number inputs and sliders are synchronized. Edit either one.</p>',
                unsafe_allow_html=True)

    params_meta = [
        ("laser_power", "Power", 100.0, 1000.0, 10.0, "W", "Laser Power", r"P_l"),
        ("scan_speed", "Speed", 100.0, 1500.0, 10.0, "mm/s", "Scan Speed", r"v_{\text{scan}}"),
        ("hatch_spacing", "Hatch", 10.0, 200.0, 5.0, "μm", "Hatch Spacing", r"l_h"),
        ("layer_thickness", "Thickness", 10.0, 100.0, 5.0, "μm", "Layer Thickness", r"l_t"),
    ]

    for key, short_label, min_v, max_v, step, unit, long_label, latex_sym in params_meta:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.number_input(
                f"{short_label} ({unit})", min_value=min_v, max_value=max_v, step=step,
                key=f"{key}_num", on_change=sync_num_to_slider, args=(key,),
                help=f"{long_label} in {unit}"
            )
        with c2:
            st.slider(
                long_label, min_value=min_v, max_value=max_v, step=step,
                key=f"{key}_slider", on_change=sync_slider_to_num, args=(key,),
                help=f"Adjust {long_label.lower()}"
            )
        val = st.session_state[f"{key}_slider"]
        st.latex(f"{latex_sym} = {val:.2f} \\text{{ {unit} }}")
        st.divider()

    # Read unified values
    P = st.session_state.laser_power_slider
    v = st.session_state.scan_speed_slider
    h_um = st.session_state.hatch_spacing_slider
    t_um = st.session_state.layer_thickness_slider

with col_results:
    st.markdown('<div class="subheader">📈 Energy Density Results</div>', unsafe_allow_html=True)

    h_mm = h_um / 1000.0
    t_mm = t_um / 1000.0

    led = P / v
    aed = P / (v * h_mm)
    ved = P / (v * h_mm * t_mm)

    st.markdown("**Current Parameter Summary**")
    st.latex(rf"P_l={P:.1f}\,\text{{W}},\; v_{{\text{{scan}}}}={v:.1f}\,\text{{mm/s}},\; l_h={h_um:.1f}\,\mu\text{{m}},\; l_t={t_um:.1f}\,\mu\text{{m}}")
    st.divider()

    # LED
    with st.container(border=True):
        st.markdown("#### 🔵 Linear Energy Density (LED)")
        st.caption("Energy deposited per unit length along a single scan track")
        st.latex(r"\text{LED} = \frac{P_l}{v_{\text{scan}}} \quad [\text{J/mm}]")
        st.latex(rf"\text{{LED}} = \frac{{{P:.1f}}}{{{v:.1f}}} \approx {led:.4f}\,\text{{J/mm}}")
        st.metric("Computed LED", f"{led:.4f} J/mm")

    # AED
    with st.container(border=True):
        st.markdown("#### 🔴 Areal Energy Density (AED)")
        st.caption("Energy deposited per unit area of the scan plane")
        st.latex(r"\text{AED} = \frac{P_l}{v_{\text{scan}} \cdot l_h} \quad [\text{J/mm}^2]")
        st.latex(rf"\text{{AED}} = \frac{{{P:.1f}}}{{{v:.1f} \times {h_mm:.4f}}} \approx {aed:.4f}\,\text{{J/mm}}^2")
        st.metric("Computed AED", f"{aed:.4f} J/mm²")

    # VED
    with st.container(border=True):
        st.markdown("#### 🟢 Volumetric Energy Density (VED)")
        st.caption("Energy deposited per unit volume of the deposited layer")
        st.latex(r"\text{VED} = \frac{P_l}{v_{\text{scan}} \cdot l_h \cdot l_t} \quad [\text{J/mm}^3]")
        st.latex(rf"\text{{VED}} = \frac{{{P:.1f}}}{{{v:.1f} \times {h_mm:.4f} \times {t_mm:.4f}}} \approx {ved:.4f}\,\text{{J/mm}}^3")
        st.metric("Computed VED", f"{ved:.4f} J/mm³")

# ============================================================
# SCHEMATIC (full width)
# ============================================================
st.markdown('<div class="subheader">📐 Process Schematic</div>', unsafe_allow_html=True)
st.plotly_chart(create_schematic(P, v, h_um, t_um), use_container_width=True)

# ============================================================
# PARAMETER REFERENCE
# ============================================================
with st.expander("📋 Parameter Reference & Typical LPBF Ranges"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🔦 Laser Power", f"{P:.1f} W",
                  help="Optical power delivered by the laser source. Higher power increases energy input.")
    with c2:
        st.metric("⚡ Scan Speed", f"{v:.1f} mm/s",
                  help="Velocity of laser beam movement. Inversely affects all energy density metrics.")
    with c3:
        st.metric("↔️ Hatch Spacing", f"{h_um:.1f} μm",
                  help="Distance between adjacent parallel scan tracks. Affects overlap and heat accumulation.")
    with c4:
        st.metric("📏 Layer Thickness", f"{t_um:.1f} μm",
                  help="Powder layer thickness per build cycle. Critical for VED and part density.")

# ============================================================
# VISUALIZATIONS
# ============================================================
st.markdown('<div class="subheader">📊 Energy Density Visualization</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📉 Trends vs. Scan Speed", "🗺️ 2D Process Window (VED)"])

with tab1:
    st.markdown("Compare how each energy density metric evolves with scanning speed. "
                "Your current speed is marked with a vertical dashed line.")
    scan_speeds = np.linspace(100, 1500, 200)
    powers = [200, 350, 500]
    colors = ["#3b82f6", "#10b981", "#f59e0b"]

    fig_trends = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Linear Energy Density  (J/mm)",
                        "Areal Energy Density  (J/mm²)",
                        "Volumetric Energy Density  (J/mm³)")
    )

    for p, color in zip(powers, colors):
        led_vals = p / scan_speeds
        aed_vals = p / (scan_speeds * h_mm)
        ved_vals = p / (scan_speeds * h_mm * t_mm)

        # Row 1: LED
        fig_trends.add_trace(go.Scatter(
            x=scan_speeds, y=led_vals, mode="lines", name=f"{p} W",
            line=dict(color=color, width=2.5),
            legendgroup=str(p), showlegend=True,
            hovertemplate="%{x:.0f} mm/s<br>%{y:.4f} J/mm<<extra></extra>"
        ), row=1, col=1)

        # Row 2: AED
        fig_trends.add_trace(go.Scatter(
            x=scan_speeds, y=aed_vals, mode="lines", name=f"{p} W",
            line=dict(color=color, width=2.5, dash="dash"),
            legendgroup=str(p), showlegend=False,
            hovertemplate="%{x:.0f} mm/s<br>%{y:.4f} J/mm²<<extra></extra>"
        ), row=2, col=1)

        # Row 3: VED
        fig_trends.add_trace(go.Scatter(
            x=scan_speeds, y=ved_vals, mode="lines", name=f"{p} W",
            line=dict(color=color, width=2.5, dash="dot"),
            legendgroup=str(p), showlegend=False,
            hovertemplate="%{x:.0f} mm/s<br>%{y:.4f} J/mm³<<extra></extra>"
        ), row=3, col=1)

    # Current-speed marker on all three rows
    fig_trends.add_vline(x=v, line=dict(color="black", width=1, dash="dot"), row=1, col=1)
    fig_trends.add_vline(x=v, line=dict(color="black", width=1, dash="dot"), row=2, col=1)
    fig_trends.add_vline(x=v, line=dict(color="black", width=1, dash="dot"), row=3, col=1)

    fig_trends.update_layout(
        height=720, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Laser Power")
    )
    fig_trends.update_xaxes(title_text="Scanning Speed (mm/s)", row=3, col=1)
    st.plotly_chart(fig_trends, use_container_width=True)

with tab2:
    st.markdown("Interactive process map showing VED as a function of laser power and scan speed. "
                "The white **×** marks your current settings.")
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
        x=[P], y=[v], mode="markers+text",
        marker=dict(color="white", size=14, symbol="x", line=dict(color="black", width=2)),
        text=["Current"], textposition="top center", name="Current Setting"
    ))
    fig2.update_layout(
        xaxis_title="Laser Power (W)", yaxis_title="Scan Speed (mm/s)",
        height=550, template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# INSIGHTS
# ============================================================
st.markdown('<div class="subheader">💡 Process Optimization Insights</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.info(f"""
    **📏 Linear Energy Density (LED)**  
    **Current:** `{led:.4f} J/mm`

    • ∝ 1 / v_scan — slower scans deposit more energy per mm  
    • Typical Al alloys: **0.25 – 0.9 J/mm**  
    • Too high → keyholing, vaporization  
    • Too low → lack of fusion, porosity
    """)

with c2:
    st.warning(f"""
    **🔲 Areal Energy Density (AED)**  
    **Current:** `{aed:.4f} J/mm²`

    • Accounts for hatch spacing (l_h)  
    • Optimal for AlSiMg: **60 – 140 J/mm²**  
    • Controls track overlap & surface finish  
    • Critical for minimizing balling defects
    """)

with c3:
    st.success(f"""
    **🧊 Volumetric Energy Density (VED)**  
    **Current:** `{ved:.4f} J/mm³`

    • Most predictive metric for bulk properties  
    • Target for >99.5% density: **55 – 95 J/mm³**  
    • Balance needed: avoid porosity vs. grain coarsening  
    • Material-specific optimization required
    """)

# ============================================================
# EXPORT & FOOTER
# ============================================================
st.markdown("---")
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    if st.button("🔄 Reset to Defaults", use_container_width=True, type="secondary"):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

with c2:
    export_data = {
        "timestamp": "2025",
        "parameters": {
            "laser_power_W": P,
            "scan_speed_mm_s": v,
            "hatch_spacing_um": h_um,
            "layer_thickness_um": t_um
        },
        "results": {
            "LED_J_per_mm": round(led, 6),
            "AED_J_per_mm2": round(aed, 6),
            "VED_J_per_mm3": round(ved, 6)
        },
        "units": {"LED": "J/mm", "AED": "J/mm²", "VED": "J/mm³"},
        "material_context": "AlSiMg1.4Zr alloy (reference)"
    }
    st.download_button(
        label="📥 Export Results (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name=f"laser_energy_{P:.0f}W_{v:.0f}mms.json",
        mime="application/json",
        use_container_width=True
    )

with c3:
    st.markdown("""
    <div style="text-align: right; color: #64748b; font-size: 0.9em; line-height: 1.5;">
    <b>📚 Reference:</b><br>
    Wang, X., Geng, Y., Oliinyk, Y., Zhang, Z., & Kunwar, A. (2025).<<br>
    <em>Multiscale Computation and Experimental Insights into Thermal History and Composition based Study of Strength-Ductility Synergy in Zr-Enhanced AlSiMg Alloys</em>.<<br><br>
    <b>⚠️ Disclaimer:</b> For research and educational purposes. Validate experimentally for production use.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    ⚡ <b>Laser Energy Density Calculator</b> | Built with Streamlit & Plotly | Optimized for AM research
</div>
""", unsafe_allow_html=True)
