import streamlit as st
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib

# Check if LaTeX is available
try:
    matplotlib.checkdep_usetex(True)
    latex_available = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{textcomp}')
except Exception:
    latex_available = False
    plt.rc('text', usetex=False)
    plt.rc('font', family='sans-serif')

# Set page configuration
st.set_page_config(page_title="AED vs Depth of Meltpool", layout="wide", page_icon="📊")

# Custom CSS styling and MathJax for LaTeX rendering in Plotly and markdown
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
    .latex-input {font-family: monospace; font-size: 1.1em;}
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

# Warning if LaTeX is unavailable
if not latex_available:
    st.warning("LaTeX is not installed or accessible. Matplotlib will use default rendering instead of LaTeX. For full LaTeX support, install a LaTeX distribution (e.g., TeX Live or MiKTeX).")

# Title
st.markdown('<div class="header">AED vs d^max_MPB Plotter</div>', unsafe_allow_html=True)
st.markdown("Visualize the relationship between **Areal Energy Density (AED)** and **Maximum Vertical Distance to Melt Pool Boundary (d^max_MPB)** with customizable plot settings. For Plotly, use LaTeX with double backslashes (e.g., $\\text{AED (J/mm}^2\\text{)}$). For Matplotlib, use single backslashes if LaTeX is available (e.g., $d^\\text{max}_\\text{MPB}$), or plain text if not.")

# Data
data = [[4.46, 123.0], [3.57, 104.0], [2.98, 101.0], [6.25, 128.0], [5.0, 119.0], [4.17, 107.0]]
aed = [row[0] for row in data]
dmax = [row[1] for row in data]

# Fit a quadratic polynomial
coeffs = np.polyfit(aed, dmax, 2)
poly = np.poly1d(coeffs)
aed_fit = np.linspace(min(aed), max(aed), 100)
dmax_fit = poly(aed_fit)

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="subheader">Plot Customization</div>', unsafe_allow_html=True)
    
    # Plotly LaTeX inputs
    st.markdown('<div class="subheader">Plotly LaTeX Labels (use \\ for MathJax)</div>', unsafe_allow_html=True)
    plotly_title = st.text_input("Plotly Plot Title (LaTeX)", r"$\text{Effect of AED on } d^{\text{max}}_{\text{MPB}}$", help="Enter title with LaTeX, e.g., $\\text{Effect of AED on } d^{\\text{max}}_{\\text{MPB}}$")
    plotly_x_label = st.text_input("Plotly X-Axis Label (LaTeX)", r"$\text{AED (J/mm}^2\text{)}$", help="Enter x-axis label with LaTeX, e.g., $\\text{AED (J/mm}^2\\text{)}$")
    plotly_y_label = st.text_input("Plotly Y-Axis Label (LaTeX)", r"$d^{\text{max}}_{\text{MPB}}$ ($\mu\text{m}$)", help="Enter y-axis label with LaTeX, e.g., $d^{\\text{max}}_{\\text{MPB}}$ ($\mu\\text{m}$)")
    plotly_point_label = st.text_input("Plotly Data Points Legend (LaTeX)", r"$\text{Data Points}$", help="Enter legend label for points with LaTeX")
    plotly_curve_label = st.text_input("Plotly Fitted Curve Legend (LaTeX)", r"$\text{Quadratic Fit: } y = ax^2 + bx + c$", help="Enter legend label for curve with LaTeX")
    
    # Plotly LaTeX preview
    st.markdown('<div class="subheader">Plotly LaTeX Preview</div>', unsafe_allow_html=True)
    st.markdown(f"**Title:** {plotly_title}", unsafe_allow_html=True)
    st.markdown(f"**X-axis:** {plotly_x_label}", unsafe_allow_html=True)
    st.markdown(f"**Y-axis:** {plotly_y_label}", unsafe_allow_html=True)
    st.markdown(f"**Points Legend:** {plotly_point_label}", unsafe_allow_html=True)
    st.markdown(f"**Curve Legend:** {plotly_curve_label}", unsafe_allow_html=True)
    
    # Matplotlib LaTeX inputs
    st.markdown(f'<div class="subheader">Matplotlib Labels {"(use single \\ for LaTeX if available, else plain text)" if not latex_available else "(use single \\ for LaTeX)"}</div>', unsafe_allow_html=True)
    mpl_title = st.text_input("Matplotlib Plot Title", r"$\text{Effect of AED on } d^\text{max}_\text{MPB}$" if latex_available else "Effect of AED on d_max_MPB", help="Enter title with LaTeX (e.g., $d^\text{max}_\text{MPB}$) if LaTeX is available, else plain text")
    mpl_x_label = st.text_input("Matplotlib X-Axis Label", r"$\text{AED (J/mm}^2\text{)}$" if latex_available else "AED (J/mm²)", help="Enter x-axis label with LaTeX (e.g., $\text{AED (J/mm}^2\text{)}$) if LaTeX is available, else plain text")
    mpl_y_label = st.text_input("Matplotlib Y-Axis Label", r"$d^\text{max}_\text{MPB}$ ($\mu\text{m}$)" if latex_available else "d_max_MPB (μm)", help="Enter y-axis label with LaTeX (e.g., $d^\text{max}_\text{MPB}$ ($\mu\text{m}$)) if LaTeX is available, else plain text")
    mpl_point_label = st.text_input("Matplotlib Data Points Legend", r"$\text{Data Points}$" if latex_available else "Data Points", help="Enter legend label for points with LaTeX if LaTeX is available, else plain text")
    mpl_curve_label = st.text_input("Matplotlib Fitted Curve Legend", r"$\text{Quadratic Fit: } y = ax^2 + bx + c$" if latex_available else "Quadratic Fit: y = ax^2 + bx + c", help="Enter legend label for curve with LaTeX if LaTeX is available, else plain text")
    
    # Matplotlib LaTeX preview
    st.markdown('<div class="subheader">Matplotlib LaTeX Preview</div>', unsafe_allow_html=True)
    st.markdown(f"**Title:** {mpl_title}", unsafe_allow_html=True)
    st.markdown(f"**X-axis:** {mpl_x_label}", unsafe_allow_html=True)
    st.markdown(f"**Y-axis:** {mpl_y_label}", unsafe_allow_html=True)
    st.markdown(f"**Points Legend:** {mpl_point_label}", unsafe_allow_html=True)
    st.markdown(f"**Curve Legend:** {mpl_curve_label}", unsafe_allow_html=True)
    
    # Font size sliders
    st.markdown('<div class="subheader">Font Sizes</div>', unsafe_allow_html=True)
    axis_label_font_size = st.slider("Axis Label Font Size (pt)", 10, 24, 14, step=1, help="Adjust font size for axis labels")
    tick_label_font_size = st.slider("Axis Tick Label Font Size (pt)", 8, 20, 12, step=1, help="Adjust font size for axis tick labels")
    
    # Customization options
    curve_color = st.color_picker("Curve Color", "#FF4B4B", help="Select color for the fitted curve")
    curve_thickness = st.slider("Curve Thickness (px)", 1.0, 5.0, 2.0, step=0.5, help="Adjust thickness of the fitted curve")
    point_color = st.color_picker("Point Color", "#1F77B4", help="Select color for data points")
    point_size = st.slider("Point Size (px)", 5.0, 20.0, 10.0, step=1.0, help="Adjust size of data points")
    legend_font_size = st.slider("Legend Font Size (pt)", 8, 20, 12, step=1, help="Adjust legend font size")
    legend_location = st.selectbox("Legend Location", ["top-left", "top-right", "bottom-left", "bottom-right"], index=1, help="Select legend position")
    axis_line_thickness = st.slider("Axis Line Thickness (px)", 1.0, 5.0, 2.0, step=0.5, help="Adjust thickness of axis lines")
    tick_length = st.slider("Tick Length (px)", 5.0, 15.0, 8.0, step=1.0, help="Adjust length of axis ticks")
    tick_width = st.slider("Tick Width (px)", 1.0, 5.0, 2.0, step=0.5, help="Adjust width of axis ticks")
    fig_width = st.slider("Figure Width (px)", 400, 1200, 800, step=50, help="Adjust figure width")
    fig_height = st.slider("Figure Height (px)", 300, 800, 500, step=50, help="Adjust figure height")

with col2:
    # Plotly Plot
    st.markdown('<div class="subheader">Plotly: Publication-Quality Plot</div>', unsafe_allow_html=True)
    fig_plotly = go.Figure()
    
    # Add scatter points
    fig_plotly.add_trace(go.Scatter(
        x=aed,
        y=dmax,
        mode="markers",
        name=plotly_point_label,
        marker=dict(color=point_color, size=point_size),
    ))
    
    # Add fitted curve
    fig_plotly.add_trace(go.Scatter(
        x=aed_fit,
        y=dmax_fit,
        mode="lines",
        name=f"{plotly_curve_label} ($y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}$)",
        line=dict(color=curve_color, width=curve_thickness),
    ))
    
    # Configure legend location
    legend_x, legend_y = {"top-left": (0.01, 0.99), "top-right": (0.99, 0.99), 
                         "bottom-left": (0.01, 0.01), "bottom-right": (0.99, 0.01)}[legend_location]
    
    # Update Plotly layout
    fig_plotly.update_layout(
        title=dict(text=plotly_title, x=0.5, xanchor="center", font=dict(size=axis_label_font_size)),
        xaxis_title=plotly_x_label,
        yaxis_title=plotly_y_label,
        width=fig_width,
        height=fig_height,
        template="plotly_white",
        font=dict(size=14),
        legend=dict(
            x=legend_x,
            y=legend_y,
            xanchor="right" if "right" in legend_location else "left",
            yanchor="top" if "top" in legend_location else "bottom",
            font=dict(size=legend_font_size),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        xaxis=dict(
            linecolor="black",
            linewidth=axis_line_thickness,
            mirror=True,
            ticks="outside",
            ticklen=tick_length,
            tickwidth=tick_width,
            showline=True,
            gridcolor="lightgrey",
            tickfont=dict(size=tick_label_font_size),
            titlefont=dict(size=axis_label_font_size)
        ),
        yaxis=dict(
            linecolor="black",
            linewidth=axis_line_thickness,
            mirror=True,
            ticks="outside",
            ticklen=tick_length,
            tickwidth=tick_width,
            showline=True,
            gridcolor="lightgrey",
            tickfont=dict(size=tick_label_font_size),
            titlefont=dict(size=axis_label_font_size)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    st.plotly_chart(fig_plotly, use_container_width=False)
    
    # Matplotlib Plot
    st.markdown('<div class="subheader">Matplotlib: Publication-Quality Plot</div>', unsafe_allow_html=True)
    
    # Create Matplotlib figure
    fig_mpl, ax = plt.subplots(figsize=(fig_width/100, fig_height/100), dpi=100)
    
    # Scatter points
    ax.scatter(aed, dmax, c=point_color, s=point_size**2, label=mpl_point_label)
    
    # Fitted curve
    ax.plot(aed_fit, dmax_fit, c=curve_color, linewidth=curve_thickness, 
            label=f"{mpl_curve_label} ($y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}$)")
    
    # Configure Matplotlib plot
    ax.set_title(mpl_title, fontsize=axis_label_font_size, pad=15)
    ax.set_xlabel(mpl_x_label, fontsize=axis_label_font_size)
    ax.set_ylabel(mpl_y_label, fontsize=axis_label_font_size)
    
    # Axis and tick settings
    ax.spines['top'].set_linewidth(axis_line_thickness)
    ax.spines['right'].set_linewidth(axis_line_thickness)
    ax.spines['left'].set_linewidth(axis_line_thickness)
    ax.spines['bottom'].set_linewidth(axis_line_thickness)
    ax.tick_params(axis='both', which='major', length=tick_length, width=tick_width, labelsize=tick_label_font_size)
    ax.grid(True, color='lightgrey', linestyle='--', alpha=0.7)
    
    # Legend
    loc_map = {"top-left": "upper left", "top-right": "upper right", 
               "bottom-left": "lower left", "bottom-right": "lower right"}
    ax.legend(loc=loc_map[legend_location], fontsize=legend_font_size, 
              frameon=True, facecolor='white', edgecolor='black')
    
    # Save Matplotlib figure to a buffer
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    st.image(buf, use_column_width=False, width=fig_width)
    
    plt.close(fig_mpl)

# Additional LaTeX annotation
st.markdown("---")
st.markdown('<div class="subheader">Fit Equation</div>', unsafe_allow_html=True)
st.markdown(r"$\text{Equation: } y = ax^2 + bx + c$", unsafe_allow_html=True)
st.markdown(fr"$\text{{Coefficients: }} a = {coeffs[0]:.4f}, b = {coeffs[1]:.4f}, c = {coeffs[2]:.4f}$", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Data based on AED and meltpool depth measurements for AlSiMg1.4Zr alloy.", unsafe_allow_html=True)
