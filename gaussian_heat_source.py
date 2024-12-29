import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

#st.title('Super-Gaussian Heat Source Visualization. The default is a Gaussian Heat Source with SGO (k) = 1.0 ')
st.title('Comparative visualization of Gaussian Heat Source with two laser power values')

# Define the colormap list globally
cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']

st.sidebar.header('Parameters')
A = st.sidebar.slider('A', min_value=0.00001, max_value=5.0, value=2.0, step=0.0001)
P1 = st.sidebar.slider('P1 (Power - Image 1)', min_value=1, max_value=5000, value=250, step=1)
P2 = st.sidebar.slider('P2 (Power - Image 2)', min_value=1, max_value=5000, value=350, step=1)
k = st.sidebar.slider('k', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
eta = st.sidebar.slider('η (Efficiency)', min_value=0.0, max_value=1.0, value=1.0, step=0.01)
r_0 = st.sidebar.slider('r₀ (Beam Radius)', min_value=20.0, max_value=500.0, value=25.0, step=0.01)
C = st.sidebar.slider('C', min_value=0.1, max_value=4.0, value=2.0, step=0.1)
colormap_index = st.sidebar.slider('Colormap', min_value=0, max_value=9, value=6, step=1)

# Function to calculate the intensity distribution
def super_gaussian_intensity(A, P, k, eta, r_0, C, x_values, y_values):
    r0 = r_0 #* 1.0e-6  # converting to micrometer
    x, y = np.meshgrid(x_values, y_values)

    r = np.sqrt(x ** 2 + y ** 2)
    #F = np.where(r0 - r < 0, 0, 1)
    F = 1.0

    intensity = F * ((A ** (1 / k) * k * P * eta) / (np.pi * r0 ** 2 * math.gamma(1 / k))) * (np.exp(-C * (r ** 2 / r0 ** 2) ** k))
    
    return intensity

# Generate x and y values for plotting
x_values = np.linspace(-50, 50, 100)
y_values = np.linspace(-50, 50, 100)


# Calculate intensity distribution for P1
intensity_P1 = super_gaussian_intensity(A, P1, k, eta, r_0, C, x_values, y_values)

# Calculate intensity distribution for P2
intensity_P2 = super_gaussian_intensity(A, P2, k, eta, r_0, C, x_values, y_values)

# Find the maximum intensity across both plots
globalmax_intensity = max(np.max(intensity_P1), np.max(intensity_P2))

# Find the maximum and minimum power values
max_laserpower = max(P1,P2)
min_laserpower = min(P1,P2)


# Display the maximum peak intensity
st.sidebar.subheader('Maximum Peak Intensity')
#st.sidebar.write(f'The maximum peak intensity at 350 W is: {globalmax_intensity:.2e} W/$\mu$m²')
st.sidebar.write(f'The maximum peak intensity at {max_laserpower}W is: {globalmax_intensity:.2e} W/$\mu$m²')

# Find the maximum intensity for 250 W
localmax_intensity = min(np.max(intensity_P1), np.max(intensity_P2))

# Display the maximum peak intensity at 250 W
#st.sidebar.write(f'The maximum peak intensity at 250 W is: {localmax_intensity:.2e} W/$\mu$m²')
st.sidebar.write(f'The maximum peak intensity at {min_laserpower}W is: {localmax_intensity:.2e} W/$\mu$m²')


# Create 3D surface plot for P1 with the same z range as the global maximum intensity
fig1 = go.Figure(data=[go.Surface(z=intensity_P1, x=x_values, y=y_values, colorscale=cmaps[colormap_index], cmin=0, cmax=globalmax_intensity)])
fig1.update_layout(scene=dict(
                        xaxis_title='X-axis', 
                        yaxis_title='Y-axis', 
                        zaxis_title='Intensity',
                        xaxis=dict(title_font=dict(size=20)),  # Adjust X-axis font size
                        yaxis=dict(title_font=dict(size=20)),  # Adjust Y-axis font size
                        zaxis=dict(title_font=dict(size=20), range=[0, globalmax_intensity]),   # Adjust Z-axis font size and range                      
                    ),
                  # annotations is also same as the title
                  #annotations=[ dict(
                  #                    showarrow=False,
                  #                    text="P = 250 W",
                  #                    font=dict(size=50),
                  #                    xref="paper",  # Refer to paper coordinates
                  #                    yref="paper",  # Refer to paper coordinates
                  #                    x=0.5,  # Horizontal position
                  #                    y=1.1   # Vertical position
                  #                    )
                  #                  ],
                  #title=dict(text="P = 250 W", font=dict(size=50), automargin=True, yref='paper'),
                  title=dict(text=f"P = {P1} W", font=dict(size=50), automargin=True, yref='paper'),
                  width=800, height=600)
fig1.update_coloraxes(colorbar=dict(
                        exponentformat='none',     # Prevents exponent format
                        thickness=20,              # Adjusts color bar thickness
                        tickfont=dict(size=15)     # Adjusts color bar tick font size
                    ))

# Show plot for P1
st.plotly_chart(fig1, use_container_width=True)

# Create 3D surface plot for P2 with the same z range as the global maximum intensity
fig2 = go.Figure(data=[go.Surface(z=intensity_P2, x=x_values, y=y_values, colorscale=cmaps[colormap_index], cmin=0, cmax=globalmax_intensity)])
fig2.update_layout(scene=dict(
                        xaxis_title='X-axis', 
                        yaxis_title='Y-axis', 
                        zaxis_title='Intensity',
                        xaxis=dict(title_font=dict(size=20)),  # Adjust X-axis font size
                        yaxis=dict(title_font=dict(size=20)),  # Adjust Y-axis font size
                        zaxis=dict(title_font=dict(size=20), range=[0, globalmax_intensity]),   # Adjust Z-axis font size and range
                    ),
                  #title=dict(text="P = 350 W", font=dict(size=50), automargin=True, yref='paper'),
                  title=dict(text=f"P = {P2} W", font=dict(size=50), automargin=True, yref='paper'),
                  width=800, height=600)
fig2.update_coloraxes(colorbar=dict(
                        exponentformat='none',     # Prevents exponent format
                        thickness=20,              # Adjusts color bar thickness
                        tickfont=dict(size=15)     # Adjusts color bar tick font size
                    ))

# Show plot for P2
st.plotly_chart(fig2, use_container_width=True)

