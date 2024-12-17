import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams

# Get all CSV files from the current folder
csv_folder = Path(__file__).parent
#csv_files = list(csv_folder.glob("*.csv"))
csv_files = sorted(csv_folder.glob("*.csv"))

if csv_files:
    # Read the data
    dataframes = []
    filenames = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Convert time to milliseconds
        if "t(s)" in df.columns:
            df["t(s)"] = df["t(s)"] * 1000.0
        dataframes.append(df)
        filenames.append(file.stem)

    # Sidebar settings
    st.sidebar.header("Plot Settings")
    fig_width = st.sidebar.slider("Figure Width (inches)", min_value=5, max_value=20, value=10)
    fig_height = st.sidebar.slider("Figure Height (inches)", min_value=5, max_value=20, value=6)
    line_thickness = st.sidebar.slider("Line Thickness", min_value=1, max_value=10, value=2)
    label_fontsize = st.sidebar.slider("Label Font Size", min_value=10, max_value=30, value=16)
    # Sidebar option for legend font size
    legend_fontsize = st.sidebar.slider("Legend Font Size", min_value=10, max_value=30, value=14)

    # Grid and Box options
    st.sidebar.header("Axes Settings")
    grid_on = st.sidebar.checkbox("Show Grid", value=True)
    grid_style = st.sidebar.selectbox("Grid Line Style", options=["--", "-", "-.", ":", "None"], index=0)
    box_on = st.sidebar.checkbox("Show Box Around Axes", value=True)

    # Allow user to edit axis labels and ticks
    x_label = st.sidebar.text_input("X-axis Label", value="Time (ms)")
    y_label = st.sidebar.text_input("Y-axis Label", value="Temperature")
    x_ticks_rotation = st.sidebar.slider("X-axis Tick Rotation", min_value=0, max_value=90, value=0)

    # Line customization
    st.sidebar.header("Line Customization")
    line_styles = ["-", "--", "-.", ":"]
    colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta", "black", "gray"]
    line_settings = {}
    for i, filename in enumerate(filenames):
        line_settings[filename] = {
            "color": st.sidebar.selectbox(f"Color for {filename}", options=colors, index=i % len(colors)),
            "style": st.sidebar.selectbox(f"Line Style for {filename}", options=line_styles, index=0),
        }

    # Allow user to edit legends dynamically
    st.sidebar.header("Edit Legends")
    legends = {}
    for i, filename in enumerate(filenames):
        legends[filename] = st.sidebar.text_input(f"Legend for {filename}", value=filename)

    # Create the plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for i, df in enumerate(dataframes):
        if "temperature" in df.columns and "t(s)" in df.columns:
            ax.plot(
                df["t(s)"],
                df["temperature"],
                label=legends[filenames[i]],
                linewidth=line_thickness,
                color=line_settings[filenames[i]]['color'],
                linestyle=line_settings[filenames[i]]['style'],
            )
        else:
            st.error(f"File {filenames[i]} is missing required columns 'temperature' and/or 't(s)'.")

    # Customize the plot
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    #ax.legend(fontsize=14, loc="best")
    ax.legend(fontsize=legend_fontsize, loc="best")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="both", width=2.5, length=8.0)
    ax.tick_params(axis="x", rotation=x_ticks_rotation)

    # Improve thickness of axes
    ax.spines['top'].set_linewidth(3.0)
    ax.spines['right'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(3.0)

    # Add grid and box
    if grid_on:
        if grid_style != "None":
            ax.grid(True, linestyle=grid_style, alpha=0.7)
    if box_on:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Display the plot
    st.pyplot(fig)
else:
    st.warning("No CSV files found in the current folder.")

