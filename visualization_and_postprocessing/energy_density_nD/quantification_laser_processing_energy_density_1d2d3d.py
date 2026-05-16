import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import json

# Set page configuration
st.set_page_config(page_title="Energy Density Calculator", layout="wide", page_icon="⚡️")

# Custom CSS styling for modern UI
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);}
    .stSlider > div > div > div > div {background-color: #4a90e2;}
    .stButton > button {
        background-color: #4a90e2; 
        color: white; 
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #357abd;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .header {
        font-size: 2.5em; 
        color: #2c3e50; 
        text-align: center; 
        margin-bottom: 20px;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 1.5em; 
        color: #34495e; 
        margin-top: 25px;
        margin-bottom: 15px;
        border-left: 4px solid #4a90e2;
        padding-left: 12px;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f4f8);
        padding: 20px; 
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
        margin-bottom: 15px;
        border: 1px solid #e0e6ed;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .formula {
        font-size: 1.15em; 
        color: #2c3e50; 
        margin: 10px 0;
        font-family: 'STIX Two Math', 'Cambria Math', 'Latin Modern Math', serif;
        line-height: 1.6;
    }
    .value {
        font-size: 1.5em; 
        color: #2980b9; 
        font-weight: 700;
        margin: 8px 0;
        font-family: monospace;
    }
    .param-label {
        font-weight: 600;
        color: #34495e;
        margin: 8px 0 4px 0;
        font-size: 1.05em;
    }
    .unit {
        color: #7f8c8d;
        font-size: 0.9em;
        font-style: italic;
        margin-left: 4px;
    }
    .schematic-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 20px 0;
    }
    .insight-box {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# KaTeX for high-quality, fast LaTeX rendering
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\\\(', right: '\\\\)', display: false},
                {left: '\\\\[', right: '\\\\]', display: true}
            ],
            throwOnError: false
        });"></script>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="header">⚡ Laser Energy Density Calculator</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #555; margin-bottom: 30px; font-size: 1.1em; max-width: 900px; margin-left: auto; margin-right: auto;">
Calculate <strong>Linear (LED)</strong>, <strong>Areal (AED)</strong>, and <strong>Volumetric (VED)</strong> 
Energy Density for laser powder bed fusion (LPBF) and additive manufacturing processes. 
Essential for optimizing melt pool dynamics, microstructure, and mechanical properties.
</div>
""", unsafe_allow_html=True)

# Layout: Input parameters | Results
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown('<div class="subheader">🔧 Input Parameters</div>', unsafe_allow_html=True)
    
    # Parameter configuration
    params = {
        'laser_power': {'default': 250.0, 'min': 100.0, 'max': 1000.0, 'step': 10.0, 'unit': 'W', 'label': 'Laser Power'},
        'scan_speed': {'default': 800.0, 'min': 100.0, 'max': 1500.0, 'step': 10.0, 'unit': 'mm/s', 'label': 'Scan Speed'},
        'hatch_spacing': {'default': 70.0, 'min': 10.0, 'max': 200.0, 'step': 5.0, 'unit': 'μm', 'label': 'Hatch Spacing'},
        'layer_thickness': {'default': 30.0, 'min': 10.0, 'max': 100.0, 'step': 5.0, 'unit': 'μm', 'label': 'Layer Thickness'}
    }
    
    # Initialize and manage session state for each parameter
    for param, config in params.items():
        if f'{param}_val' not in st.session_state:
            st.session_state[f'{param}_val'] = config['default']
        
        st.markdown(f'<div class="param-label">{config["label"]} <span class="unit">({config["unit"]})</span></div>', unsafe_allow_html=True)
        
        # Dual input: number input + slider in same row
        col_num, col_sld = st.columns([1, 3])
        with col_num:
            num_val = st.number_input(
                "",
                min_value=config['min'],
                max_value=config['max'],
                value=st.session_state[f'{param}_val'],
                step=config['step'],
                key=f"{param}_num",
                label_visibility="collapsed"
            )
        with col_sld:
            slide_val = st.slider(
                "",
                min_value=config['min'],
                max_value=config['max'],
                value=st.session_state[f'{param}_val'],
                step=config['step'],
                key=f"{param}_sld",
                label_visibility="collapsed"
            )
        
        # Sync: use whichever was most recently changed (simple heuristic)
        new_val = slide_val if slide_val != st.session_state[f'{param}_val'] else num_val
        st.session_state[f'{param}_val'] = new_val
        st.session_state[param] = new_val
        
        st.markdown("---")

with col2:
    st.markdown('<div class="subheader">📊 Calculated Energy Densities</div>', unsafe_allow_html=True)
    
    # Extract and convert parameters
    P = st.session_state.laser_power          # W
    v = st.session_state.scan_speed           # mm/s
    h = st.session_state.hatch_spacing / 1000 # μm → mm
    t = st.session_state.layer_thickness / 1000 # μm → mm
    
    # Core calculations
    led = P / v                    # J/mm
    aed = P / (v * h)              # J/mm²
    ved = P / (v * h * t)          # J/mm³
    
    # Results configuration with LaTeX formulas
    metrics = [
        {
            'name': 'Linear Energy Density (LED)',
            'symbol': r'\text{LED}',
            'formula': r'\frac{P_{\text{l}}}{v_{\text{scan}}}',
            'substitution': rf'\frac{{{P:.1f}}}{{{v:.1f}}}',
            'value': led,
            'unit': r'\text{J/mm}',
            'desc': 'Energy deposited per unit length along the scan direction'
        },
        {
            'name': 'Areal Energy Density (AED)',
            'symbol': r'\text{AED}',
            'formula': r'\frac{P_{\text{l}}}{v_{\text{scan}} \cdot l_{\text{h}}}',
            'substitution': rf'\frac{{{P:.1f}}}{{{v:.1f} \cdot {h:.4f}}}',
            'value': aed,
            'unit': r'\text{J/mm}^2',
            'desc': 'Energy per unit area of the scanned surface layer'
        },
        {
            'name': 'Volumetric Energy Density (VED)',
            'symbol': r'\text{VED}',
            'formula': r'\frac{P_{\text{l}}}{v_{\text{scan}} \cdot l_{\text{h}} \cdot l_{\text{t}}}',
            'substitution': rf'\frac{{{P:.1f}}}{{{v:.1f} \cdot {h:.4f} \cdot {t:.4f}}}',
            'value': ved,
            'unit': r'\text{J/mm}^3',
            'desc': 'Energy per unit volume of processed material (most comprehensive metric)'
        }
    ]
    
    # Display each metric with LaTeX rendering
    for m in metrics:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**{m['name']}**")
            # LaTeX formula with substitution
            st.markdown(
                f'<div class="formula">'
                f'${m["symbol"]} = {m["formula"]} = {m["substitution"]} \\approx {m["value"]:.4f}$'
                f'</div>', 
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="value">💡 {m["value"]:.4f} {m["unit"]}</div>', unsafe_allow_html=True)
            st.markdown(f'*{m["desc"]}*')
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SCHEMATIC REPRESENTATION: Rectangular Slab Geometry
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="subheader">🔬 Schematic: Rectangular Slab Geometry</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background: white; border-radius: 12px; padding: 15px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 10px 0;">
<strong>Visual guide to laser processing parameters:</strong> 
Laser beam scans across powder bed with defined <em>hatch spacing</em> between parallel tracks, 
building up layers of thickness <em>l<sub>t</sub></em>. Energy density metrics quantify energy distribution.
</div>
""", unsafe_allow_html=True)

def create_schematic(P, v, h_um, t_um):
    """Create interactive Plotly schematic showing laser processing geometry"""
    h_mm = h_um / 1000
    t_mm = t_um / 1000
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('📐 3D Isometric View', '🔍 Top View (XY Plane)'),
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}]]
    )
    
    # ===== 3D VIEW (Left) =====
    # Draw rectangular slab (simplified wireframe)
    # Bottom face
    x_base = [0, 10, 10, 0, 0]
    y_base = [0, 0, 6, 6, 0]
    z_base = [0, 0, 0, 0, 0]
    fig.add_trace(go.Scatter3d(
        x=x_base, y=y_base, z=z_base,
        mode='lines', name='Slab Boundary',
        line=dict(color='#3498db', width=3),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)
    
    # Top face (offset by layer thickness)
    z_top = [t_mm*15]*5  # exaggerated for visibility
    fig.add_trace(go.Scatter3d(
        x=x_base, y=y_base, z=z_top,
        mode='lines', line=dict(color='#3498db', width=3),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)
    
    # Vertical edges
    for i in range(4):
        fig.add_trace(go.Scatter3d(
            x=[x_base[i], x_base[i]],
            y=[y_base[i], y_base[i]],
            z=[0, z_top[i]],
            mode='lines', line=dict(color='#3498db', width=2),
            hoverinfo='skip', showlegend=False
        ), row=1, col=1)
    
    # Laser scan path (red line on top surface)
    scan_y = 3
    fig.add_trace(go.Scatter3d(
        x=[1, 9], y=[scan_y, scan_y], z=[z_top[0]+0.3, z_top[0]+0.3],
        mode='lines+markers', name='Laser Scan Path',
        line=dict(color='#e74c3c', width=4),
        marker=dict(size=5, color='#e74c3c'),
        hovertemplate='<b>Laser Beam</b><br>Power: %{text} W<extra></extra>',
        text=[P]*2
    ), row=1, col=1)
    
    # Scan direction arrow
    fig.add_trace(go.Scatter3d(
        x=[9, 9.8], y=[scan_y, scan_y], z=[z_top[0]+0.3, z_top[0]+0.3],
        mode='lines', name='Scan Direction →',
        line=dict(color='#2ecc71', width=3, dash='dot'),
        showlegend=True
    ), row=1, col=1)
    
    # Hatch spacing indicator (parallel dashed line)
    fig.add_trace(go.Scatter3d(
        x=[1, 9], y=[scan_y + h_mm*8, scan_y + h_mm*8], z=[z_top[0]+0.3, z_top[0]+0.3],
        mode='lines', name='Adjacent Scan Track',
        line=dict(color='#9b59b6', width=2, dash='dash'),
        showlegend=True
    ), row=1, col=1)
    
    # Annotations for 3D view
    fig.add_annotation(
        x=5, y=scan_y, z=z_top[0]+1.2,
        text=f"<b>P<sub>l</sub></b> = {P} W<br><span style='font-size:0.9em'>Laser Power</span>",
        showarrow=True, arrowhead=2, ax=0, ay=-50, az=0,
        font=dict(size=10, color='#e74c3c'), align='center',
        row=1, col=1
    )
    fig.add_annotation(
        x=9.5, y=scan_y+0.5, z=z_top[0]+0.3,
        text=f"<b>v<sub>scan</sub></b><br>{v} mm/s",
        showarrow=True, arrowhead=2, ax=30, ay=-20, az=0,
        font=dict(size=10, color='#2ecc71'),
        row=1, col=1
    )
    
    # ===== TOP VIEW (Right) =====
    # Draw multiple scan lines to show hatch pattern
    n_lines = 4
    for i in range(n_lines):
        y_pos = 1 + i * (h_mm * 7)  # scaled for visibility
        fig.add_trace(go.Scatter(
            x=[1, 9], y=[y_pos, y_pos],
            mode='lines', name='Scan Track' if i==0 else '',
            line=dict(color='#e74c3c', width=2.5),
            hovertemplate='<b>Scan Track</b><br>Y: %{y:.2f} mm<extra></extra>',
            showlegend=(i==0)
        ), row=1, col=2)
    
    # Hatch spacing dimension line
    y1, y2 = 1, 1 + h_mm*7
    fig.add_trace(go.Scatter(
        x=[9.3, 9.3], y=[y1, y2],
        mode='lines', name='Hatch Spacing',
        line=dict(color='#9b59b6', width=2, dash='dot'),
        showlegend=True
    ), row=1, col=2)
    # Arrowheads for dimension
    fig.add_trace(go.Scatter(
        x=[9.3, 9.15, 9.3, 9.45], y=[y1, y1+0.15, y1, y1+0.15],
        mode='lines', line=dict(color='#9b59b6', width=1),
        hoverinfo='skip', showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[9.3, 9.15, 9.3, 9.45], y=[y2, y2-0.15, y2, y2-0.15],
        mode='lines', line=dict(color='#9b59b6', width=1),
        hoverinfo='skip', showlegend=False
    ), row=1, col=2)
    
    # Hatch spacing label
    fig.add_annotation(
        x=10.1, y=(y1+y2)/2,
        text=f"<b>l<sub>h</sub></b><br>{h_um:.0f} μm",
        showarrow=False,
        font=dict(size=11, color='#9b59b6'),
        align='left', bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#9b59b6', borderwidth=1, borderpad=4,
        row=1, col=2
    )
    
    # Layer thickness indicator (bottom annotation)
    fig.add_annotation(
        x=5, y=-0.8,
        text=f"<b>l<sub>t</sub></b> = {t_um:.0f} μm<br><span style='font-size:0.9em;color:#666'>Layer Thickness</span>",
        showarrow=False,
        font=dict(size=11, color='#34495e'),
        align='center',
        row=1, col=2
    )
    
    # Layout configuration
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=15, t=60, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        hoverlabel=dict(bgcolor="white", font_size=11)
    )
    
    # 3D scene settings
    fig.update_scenes(
        xaxis=dict(visible=False, range=[-1, 11]),
        yaxis=dict(visible=False, range=[-1, 8]),
        zaxis=dict(visible=False, range=[0, max(z_top)+1]),
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
        bgcolor='rgba(248,249,250,0.5)',
        row=1, col=1
    )
    
    # 2D plot settings
    fig.update_xaxes(title_text="X Direction (mm)", row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="Y Direction (mm)", row=1, col=2, scaleanchor="x", scaleratio=1, showgrid=True)
    
    return fig

# Render schematic
schematic_fig = create_schematic(P, v, st.session_state.hatch_spacing, st.session_state.layer_thickness)
st.plotly_chart(schematic_fig, use_container_width=True)

# Parameter summary cards
with st.expander("📋 Parameter Reference & Units", expanded=False):
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric("🔦 Laser Power", f"{P:.1f} W", 
                 help="Optical power delivered by the laser source. Higher power increases energy input.")
    with col_p2:
        st.metric("⚡ Scan Speed", f"{v:.1f} mm/s", 
                 help="Velocity of laser beam movement. Inversely affects all energy density metrics.")
    with col_p3:
        st.metric("↔️ Hatch Spacing", f"{st.session_state.hatch_spacing:.1f} μm", 
                 help="Distance between adjacent parallel scan tracks. Affects overlap and heat accumulation.")
    with col_p4:
        st.metric("📏 Layer Thickness", f"{st.session_state.layer_thickness:.1f} μm", 
                 help="Powder layer thickness per build cycle. Critical for VED and part density.")

# ═══════════════════════════════════════════════════════════════
# INTERACTIVE VISUALIZATION: Energy Density Trends
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="subheader">📈 Energy Density Trends Analysis</div>', unsafe_allow_html=True)
st.markdown("Compare how energy densities evolve with scanning speed across multiple laser power settings.")

# Generate parametric data
scan_speeds = np.linspace(200, 1500, 200)
power_levels = [200, 350, 500]  # W

# Create tabbed interface for each density type
tab_led, tab_aed, tab_ved = st.tabs(["📏 Linear (LED)", "🔲 Areal (AED)", "🧊 Volumetric (VED)"])

def create_trend_plot(y_values_dict, y_label, y_unit, title_suffix):
    """Helper to create consistent trend plots"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, p in enumerate(power_levels):
        y_vals = y_values_dict[p]
        fig.add_trace(go.Scatter(
            x=scan_speeds, y=y_vals, mode='lines', name=f'{p} W',
            line=dict(color=colors[idx % len(colors)], width=3),
            hovertemplate='<b>%{x:.0f} mm/s</b><br>%{y:.4f} ' + y_unit + '<extra>' + f'{p} W</extra>'
        ))
    
    fig.update_layout(
        title=f'Energy Density vs. Scanning Speed {title_suffix}',
        xaxis_title='Scanning Speed (mm/s)',
        yaxis_title=f'{y_label} ({y_unit})',
        template='plotly_white',
        height=420,
        hovermode='x unified',
        legend_title='Laser Power',
        font=dict(size=11)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    return fig

with tab_led:
    led_data = {p: p / scan_speeds for p in power_levels}
    st.plotly_chart(create_trend_plot(led_data, 'LED', 'J/mm', '(Linear)'), use_container_width=True)
    st.caption("💡 *LED decreases hyperbolically with scan speed. Critical for melt pool depth control.*")

with tab_aed:
    aed_data = {p: p / (scan_speeds * h) for p in power_levels}
    st.plotly_chart(create_trend_plot(aed_data, 'AED', 'J/mm²', '(Areal)'), use_container_width=True)
    st.caption("💡 *AED incorporates hatch spacing. Key for surface roughness and track overlap optimization.*")

with tab_ved:
    ved_data = {p: p / (scan_speeds * h * t) for p in power_levels}
    st.plotly_chart(create_trend_plot(ved_data, 'VED', 'J/mm³', '(Volumetric)'), use_container_width=True)
    st.caption("💡 *VED is the most comprehensive metric. Directly correlates with part density and microstructure.*")

# ═══════════════════════════════════════════════════════════════
# KEY INSIGHTS & GUIDANCE
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="subheader">💡 Process Optimization Insights</div>', unsafe_allow_html=True)
col_ins1, col_ins2, col_ins3 = st.columns(3)

with col_ins1:
    st.info(f"""
    **📏 Linear Energy Density**  
    **Current:** `{led:.4f} J/mm`  
    • ∝ 1/`v_scan` — slower scans increase energy per mm  
    • Target range for Al alloys: **0.25–0.9 J/mm**  
    • Too high: keyholing, vaporization  
    • Too low: lack of fusion, porosity
    """)

with col_ins2:
    st.warning(f"""
    **🔲 Areal Energy Density**  
    **Current:** `{aed:.4f} J/mm²`  
    • Accounts for hatch strategy (`l_h`)  
    • Optimal for AlSiMg: **60–140 J/mm²**  
    • Controls track overlap & surface finish  
    • Critical for minimizing balling defects
    """)

with col_ins3:
    st.success(f"""
    **🧊 Volumetric Energy Density**  
    **Current:** `{ved:.4f} J/mm³`  
    • Most predictive of bulk properties  
    • Target for >99.5% density: **55–95 J/mm³**  
    • Balance needed: avoid porosity vs. grain coarsening  
    • Material-specific optimization required
    """)

# ═══════════════════════════════════════════════════════════════
# EXPORT & UTILITIES
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])

with col_exp1:
    if st.button("🔄 Reset to Defaults", use_container_width=True, type="secondary"):
        for param, config in params.items():
            st.session_state[f'{param}_val'] = config['default']
            st.session_state[param] = config['default']
        st.rerun()

with col_exp2:
    # Prepare export data
    export_data = {
        "timestamp": "2025",
        "parameters": {
            "laser_power_W": P,
            "scan_speed_mm_s": v,
            "hatch_spacing_um": st.session_state.hatch_spacing,
            "layer_thickness_um": st.session_state.layer_thickness
        },
        "results": {
            "LED_J_per_mm": round(led, 6),
            "AED_J_per_mm2": round(aed, 6),
            "VED_J_per_mm3": round(ved, 6)
        },
        "units": {
            "LED": "J/mm",
            "AED": "J/mm²", 
            "VED": "J/mm³"
        },
        "material_context": "AlSiMg1.4Zr alloy (reference)"
    }
    
    st.download_button(
        label="📥 Export Results (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name=f"laser_energy_density_{P}W_{v}mms.json",
        mime="application/json",
        use_container_width=True
    )

with col_exp3:
    st.markdown("""
    <div style="text-align: right; color: #666; font-size: 0.95em; line-height: 1.5;">
    <strong>📚 Academic Reference:</strong><br>
    Wang, X., Geng, Y., Oliinyk, Y., Zhang, Z., & Kunwar, A. (2025). 
    <em>Multiscale Computation and Experimental Insights into Thermal History and Composition based Study of Strength-Ductility Synergy in Zr-Enhanced AlSiMg Alloys</em>.<br><br>
    <strong>⚠️ Disclaimer:</strong> For research and educational purposes. 
    Validate parameters experimentally for production use.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9em; padding: 25px 0 10px 0; border-top: 1px solid #eee; margin-top: 10px;">
⚡ <strong>Laser Energy Density Calculator</strong> | Built with Streamlit & Plotly | 
Optimized for additive manufacturing research
</div>
""", unsafe_allow_html=True)
