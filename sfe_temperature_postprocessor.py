import streamlit as st
import numpy as np
import pandas as pd

# Function to compute the desired metrics
def compute_metrics(data1, a0):
    bur = a0 * np.sqrt(3.) / 2.  # magnitude of the Burgers vector

    nx, ny = data1.shape
    interval = 1
    stress = np.abs(data1[0:nx:interval, 1]) - np.abs(data1[0, 1])
    disp = data1[0:nx:interval, 0]

    dx = data1[1, 0] - data1[0, 0]
    slope = np.abs(np.gradient(np.abs(data1[0:nx:interval, 1])) / dx) / bur * 0.01

    peierls = [0, 0]
    target = [0, 0]
    lefthalf = nx // 2
    peierls[0], target[0] = np.max(np.abs(slope[0:lefthalf])), np.argmax(np.abs(slope[0:lefthalf]))
    peierls[1], target[1] = np.max(np.abs(slope[lefthalf:nx])), np.argmax(np.abs(slope[lefthalf:nx]))
    target[1] += lefthalf

    peierls_disp = disp[target]
    peierls_slope = slope[target]
    usfe = np.max(data1[:, 1])
    iss = max(peierls_slope[0], peierls_slope[1])

    return peierls_disp, peierls_slope, usfe, iss

# Streamlit app
st.title("Unstable Stacking Fault Energy and Ideal Shear Strength Calculator for Al-2.5Mg1.5Zr Alloy")

# Section for uploading T-a0 data
st.sidebar.header("Upload Temperature-Lattice Parameter Data")
uploaded_ta0_file = st.sidebar.file_uploader("Upload CSV file for T-a0 data", type="csv")

# Default lattice parameter
default_a0 = 4.046

if uploaded_ta0_file:
    ta0_data = pd.read_csv(uploaded_ta0_file)
    st.sidebar.write("Uploaded T-a0 data:")
    st.sidebar.write(ta0_data)

    # Get user input for temperature
    selected_temp = st.sidebar.number_input("Enter temperature (T):", min_value=float(ta0_data["T"].min()), max_value=float(ta0_data["T"].max()), step=1.0)
    
    # Fetch corresponding a0
    matching_row = ta0_data[ta0_data["T"] == selected_temp]
    if not matching_row.empty:
        a0 = matching_row.iloc[0]["a0"]
        st.sidebar.write(f"Using lattice parameter a0 = {a0:.4f}")
    else:
        st.sidebar.error("Temperature not found in the data.")
        a0 = default_a0
else:
    st.sidebar.warning("No T-a0 data uploaded. Using default a0 = 4.046.")
    a0 = default_a0

# Section for uploading main data
st.header("Upload Displacement-Stress Data")
uploaded_file = st.file_uploader("Upload your data file", type=["dat", "txt", "csv"])

if uploaded_file is not None:
    try:
        data1 = np.loadtxt(uploaded_file)
        peierls_disp, peierls_slope, usfe, iss = compute_metrics(data1, a0)

        st.write("### Results")
        st.write(f"Peierls Displacement: {peierls_disp}")
        st.write(f"Peierls Slope: {peierls_slope}")
        st.write(f"Unstable Stacking Fault Energy (USFE): {usfe}")
        st.write(f"Ideal Shear Strength (ISS): {iss}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Upload a displacement-stress data file to compute metrics.")

