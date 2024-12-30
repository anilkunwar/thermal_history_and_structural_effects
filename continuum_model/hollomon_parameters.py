import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Streamlit App
st.title("Finding average values for data of Hollomon Parameters")

# Load MathJax
st.markdown("""
    <script type="text/javascript" async 
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
""", unsafe_allow_html=True)

# Introduction to Hollomon Parameters
st.markdown("""
### What are Hollomon Parameters?
The **Hollomon equation** describes the relationship between stress and strain in materials as follows:
$$
\\sigma_H = \\sigma_0 \\cdot \\varepsilon^n
$$

where,
-  $\sigma_H$: **True stress**  
-  $\sigma_0$ : **Strength coefficient**  
-  $\\varepsilon$ : **True strain**  
-  $n$ : **Strain hardening exponent**

During laser processing, the temperature distribution is non-uniform in the AlMgSiZr alloy sample. The Hollomon parameters are temperature dependent.
After the experimental dataset is uploaded as a csv file, this app calculates the average values of the strength coefficient \\($\sigma_0$) and the strain hardening exponent \\($n$) are computed at a given temperature.
Then the visualization is done for $\sigma_0$-T and $n$-T.
""")



# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    try:
        # Calculate averages for sigma_0 (s01, s02, s03) and n (n1, n2, n3)
        data['savg'] = data[['s01', 's02', 's03']].mean(axis=1)
        data['navg'] = data[['n1', 'n2', 'n3']].mean(axis=1)

        # Display the updated DataFrame
        st.write("Data with Averages:")
        st.write(data)

        # Create Plotly plot for savg (Strength Coefficient)
        fig_savg = go.Figure()
        fig_savg.add_trace(go.Scatter(
            x=data['T(oC)'],
            y=data['savg'],
            mode="lines+markers",
            name=r'\u03C3\u2090 (Average Strength Coefficient)'  # Unicode for sigma (σ) and subscript 0 (₀)
        ))

        # Create Plotly plot for navg (Strain Hardening Exponent)
        fig_navg = go.Figure()
        fig_navg.add_trace(go.Scatter(
            x=data['T(oC)'],
            y=data['navg'],
            mode="lines+markers",
            name=r'n (Average Strain Hardening Exponent)'  # Unicode for n
        ))

        # Update layout for the plots with Unicode-style subscripts and superscripts
        fig_savg.update_layout(
            title=r'\u03C3\u2090 vs. Temperature (°C)',  # Unicode for sigma (σ) and subscript 0 (₀)
            xaxis_title=r'Temperature (\u00B0C)',  # Degree symbol (°)
            yaxis_title=r'\u03C3\u2090 (MPa)',  # Unicode for sigma (σ) and subscript 0 (₀)
            font=dict(size=16)
        )

        fig_navg.update_layout(
            title=r'n vs. Temperature (°C)',  # Unicode for n
            xaxis_title=r'Temperature (\u00B0C)',  # Degree symbol (°)
            yaxis_title=r'n',  # Unicode for n
            font=dict(size=16)
        )

        # Display the plots
        st.plotly_chart(fig_savg, use_container_width=True)
        st.plotly_chart(fig_navg, use_container_width=True)

        # Plot the results of statistical calculations using Plotly Express
        fig_savg = px.line(data, x="T(oC)", y="savg", title="Average Strength Coefficient (\(s_{\text{avg}}\)) vs. Temperature",
                           labels={"T(oC)": "Temperature (°C)", "savg": "Average Strength Coefficient (\(s_{\text{avg}}\))"})
        fig_navg = px.line(data, x="T(oC)", y="navg", title="Average Strain Hardening Exponent (\(n_{\text{avg}}\)) vs. Temperature",
                           labels={"T(oC)": "Temperature (°C)", "navg": "Average Strain Hardening Exponent (\(n_{\text{avg}}\))"})

        # Display the plots
        st.plotly_chart(fig_savg)
        st.plotly_chart(fig_navg)

        # Allow user to download the updated DataFrame
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Updated CSV", data=csv, file_name="updated_data.csv", mime="text/csv")

    except KeyError as e:
        st.error(f"Error: Missing columns in the uploaded file. Ensure your file has columns: s01, s02, s03, n1, n2, n3.")

