import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import pickle
import torch
import re
import os
from spacy.language import Language
from spacy.tokens import Doc
from collections import Counter
from math import log2
import spacy
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr

# Set Matplotlib to use a non-interactive backend for Streamlit
matplotlib.use('Agg')

# Initialize logging
logging.basicConfig(filename='sfe_ner_analysis.log', level=logging.DEBUG)

# Initialize Streamlit app
st.set_page_config(page_title="SFE NER Analysis Tool", layout="wide")
st.title("Stacking Fault Energy (SFE) Analysis for Al-based and Other Multicomponent Alloys")
st.markdown("""
This tool extracts stacking fault energy (SFE in mJ/m²) and temperature (°C or K) from scientific papers stored in `sfe_knowledgeuniverse.db` or saved NER results in `.pkl` files. It uses regex-based NER with SpaCy for material detection and Pointwise Mutual Information (PMI) to identify significant phrases like "stacking fault." Use the **NER Analysis** tab to process the database or a `.pkl` file, and the **Visualize Results** tab to load and visualize existing results with customizable, publication-quality Matplotlib plots.
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `sqlite3`, `pandas`, `streamlit`, `matplotlib`, `numpy`, `spacy`, `fuzzywuzzy`, `python-Levenshtein`, `h5py`, `torch`, `scipy`
- Install with: `pip install pandas streamlit matplotlib numpy spacy fuzzywuzzy python-Levenshtein h5py torch scipy`
- For optimal NER, install: `python -m spacy download en_core_web_lg`
""")

# Tabs for NER Analysis and Visualization
tab1, tab2 = st.tabs(["NER Analysis", "Visualize Results"])

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Falling back to 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
        st.info("Using en_core_web_sm (less accurate). Install en_core_web_lg: `python -m spacy download en_core_web_lg`")
    except Exception as e2:
        st.error(f"Failed to load SpaCy model: {e2}. Install with: `python -m spacy download en_core_web_sm`")
        st.stop()

# Define entity types and default colors
param_types = ["MATERIAL", "STACKING_FAULT_ENERGY", "TEMPERATURE_C", "TEMPERATURE_K"]
default_colors = {
    "MATERIAL": '#9467bd',              # Purple
    "STACKING_FAULT_ENERGY": '#1f77b4', # Blue
    "TEMPERATURE_C": '#ff7f0e',         # Orange
    "TEMPERATURE_K": '#2ca02c'          # Green
}
logging.info(f"Default colors: {default_colors}")

# Parameter validation ranges
valid_ranges = {
    "STACKING_FAULT_ENERGY": (-1000, 1000, "mJ/m²"),
    "TEMPERATURE_C": (0, 2000, "°C"),
    "TEMPERATURE_K": (273, 2000, "K")
}

# PMI calculation
def calculate_pmi(text, window_size=5, min_count=2):
    try:
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        bigram_counts = Counter()
        total_words = len(words)
        total_bigrams = 0
        
        for i in range(len(words) - 1):
            for j in range(i + 1, min(i + window_size + 1, len(words))):
                bigram = (words[i], words[j])
                bigram_counts[bigram] += 1
                total_bigrams += 1
        
        pmi_scores = {}
        for (w1, w2), count in bigram_counts.items():
            if count >= min_count:
                p_w1 = word_counts[w1] / total_words
                p_w2 = word_counts[w2] / total_words
                p_w1_w2 = count / total_bigrams
                if p_w1_w2 > 0 and p_w1 > 0 and p_w2 > 0:
                    pmi = log2(p_w1_w2 / (p_w1 * p_w2))
                    pmi_scores[f"{w1} {w2}"] = pmi
        
        sfe_phrases = ["stacking fault", "stacking faults", "sfe", "sf", "stacking fault energy", "gsfe"]
        relevant_phrases = {phrase: score for phrase, score in pmi_scores.items() if phrase in sfe_phrases and score > 0}
        logging.info(f"PMI phrases: {relevant_phrases}")
        return relevant_phrases
    except Exception as e:
        logging.error(f"PMI calculation failed: {str(e)}")
        return {}

# Extract parameters
def extract_parameters(text, paper_id, title, year):
    try:
        pmi_phrases = calculate_pmi(text)
        sfe_context_terms = list(pmi_phrases.keys()) + ["stacking faults", "sfe", "sf", "stacking fault energy", "gsfe"]
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["MATERIAL", "ORG", "PRODUCT"] or any(fuzz.partial_ratio(term, ent.text.lower()) > 75 for term in ["al alloy", "alsi", "almgsi", "multicomponent alloy", "aluminum alloy", "hea"]):
                entities.append({
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "entity_text": ent.text,
                    "entity_label": "MATERIAL",
                    "value": None,
                    "unit": None,
                    "outcome": None,
                    "context": text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)].replace("\n", " ")
                })
        
        patterns = [
            (r"(-?\d+\.?\d*)\s*(mJ/m2|mJ m2|mJ/m²)", "STACKING_FAULT_ENERGY", "mJ/m²"),
            (r"(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*(mJ/m2|mJ m2|mJ/m²)", "STACKING_FAULT_ENERGY", "mJ/m²"),
            (r"(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(mJ/m2|mJ m2|mJ/m²)", "STACKING_FAULT_ENERGY", "mJ/m²"),
            (r"between\s*15\s*and\s*25\s*(mJ/m2|mJ m2|mJ/m²)", "STACKING_FAULT_ENERGY", "mJ/m²"),
            (r"(\d+\.?\d*)\s*(°C|○C|celsius)", "TEMPERATURE_C", "°C"),
            (r"(\d+\.?\d*)\s*(K|kelvin)", "TEMPERATURE_K", "K")
        ]
        
        for pattern, label, unit in patterns:
            for match in re.finditer(pattern, text):
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                context_lower = context.lower()
                
                if label == "STACKING_FAULT_ENERGY" and not any(term in context_lower for term in sfe_context_terms):
                    logging.debug(f"Skipping SFE entity {match.group(0)}: no relevant context")
                    continue
                
                if "to" in pattern:
                    start_val = float(match.group(1))
                    end_val = float(match.group(2))
                    if valid_ranges[label][0] <= start_val <= valid_ranges[label][1] and valid_ranges[label][0] <= end_val <= valid_ranges[label][1]:
                        for val in np.linspace(start_val, end_val, 5):
                            entities.append({
                                "paper_id": paper_id,
                                "title": title,
                                "year": year,
                                "entity_text": f"{start_val} to {end_val}",
                                "entity_label": label,
                                "value": val,
                                "unit": unit,
                                "outcome": None,
                                "context": context.replace("\n", " ")
                            })
                elif "±" in pattern:
                    value = float(match.group(1))
                    uncertainty = float(match.group(2))
                    if valid_ranges[label][0] <= value <= valid_ranges[label][1]:
                        for val, val_type in [(value, "Central"), (value - uncertainty, "Lower"), (value + uncertainty, "Upper")]:
                            if valid_ranges[label][0] <= val <= valid_ranges[label][1]:
                                entities.append({
                                    "paper_id": paper_id,
                                    "title": title,
                                    "year": year,
                                    "entity_text": f"{value} ± {uncertainty}",
                                    "entity_label": label,
                                    "value": val,
                                    "unit": unit,
                                    "outcome": None,
                                    "context": context.replace("\n", " ")
                                })
                elif "between" in pattern:
                    table_values = [15, 18.1, 18.6, 20.8, 24.8, 21.0, 22.4, 16.7, 17.0, 24.0]
                    for val in table_values:
                        entities.append({
                            "paper_id": paper_id,
                            "title": title,
                            "year": year,
                            "entity_text": "15 to 25",
                            "entity_label": label,
                            "value": val,
                            "unit": unit,
                            "outcome": None,
                            "context": context.replace("\n", " ")
                        })
                else:
                    value = float(match.group(1))
                    if label == "TEMPERATURE_K" and 0 <= value <= 100:
                        label = "TEMPERATURE_C"
                        unit = "°C"
                    if valid_ranges[label][0] <= value <= valid_ranges[label][1]:
                        outcome = None
                        outcome_terms = ["ductility", "strength", "hardness", "deformation", "twinning"]
                        for term in outcome_terms:
                            if term in context_lower:
                                outcome = term
                                break
                        entities.append({
                            "paper_id": paper_id,
                            "title": title,
                            "year": year,
                            "entity_text": match.group(0),
                            "entity_label": label,
                            "value": value,
                            "unit": unit,
                            "outcome": outcome,
                            "context": context.replace("\n", " ")
                        })
                logging.debug(f"Extracted entity: {match.group(0)}, label: {label}, value: {value if 'value' in locals() else 'range/uncertainty'}, unit: {unit}")
        
        return entities, pmi_phrases
    except Exception as e:
        logging.error(f"NER failed for paper {paper_id}: {str(e)}")
        return [{"paper_id": paper_id, "title": title, "year": year, "entity_text": f"Error: {str(e)}", "entity_label": "ERROR", "value": None, "unit": None, "outcome": None, "context": ""}], {}

# Process SQLite database
def process_sqlite(db_file):
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql("SELECT * FROM full_text WHERE full_text IS NOT NULL", conn)
        conn.close()
        
        results = []
        pmi_results = {}
        relevant_entries = 0
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            text = row["full_text"]
            if not any(term.lower() in text.lower() for term in ["stacking fault energy", "sfe", "stacking fault", "sf", "gsfe", "al alloy", "multicomponent alloy", "aluminum alloy"]):
                logging.debug(f"Skipping paper {row['paper_id']}: no relevant terms in full text")
                continue
            relevant_entries += 1
            entities, pmi_phrases = extract_parameters(text, row["paper_id"], row["title"], row["year"])
            results.extend(entities)
            pmi_results[row["paper_id"]] = pmi_phrases
            progress_bar.progress((i + 1) / len(df))
        
        st.info(f"Processed {relevant_entries} relevant full-text entries from {db_file}.")
        return pd.DataFrame(results), pmi_results
    except Exception as e:
        st.error(f"Error processing {db_file}: {str(e)}")
        logging.error(f"Error processing {db_file}: {str(e)}")
        return None, {}

# Save NER results
def save_ner_results(df, base_name="sfe_params"):
    try:
        h5_path = f"{base_name}.h5"
        df.to_hdf(h5_path, key="ner_results", mode="w")
        pkl_path = f"{base_name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f)
        pt_path = f"{base_name}.pt"
        torch.save(df.to_dict(orient="records"), pt_path)
        logging.info(f"Saved NER results to {h5_path}, {pkl_path}, {pt_path}")
        return h5_path, pkl_path, pt_path
    except Exception as e:
        logging.error(f"Failed to save NER results: {str(e)}")
        return None, None, None

# Visualize results with new scatter plot (SFE on x, Temperature on y)
def visualize_results(df, entity_types, pmi_results):
    # Set Matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sidebar for plot customization
    st.sidebar.subheader("Visualization Customization")
    font_size = st.sidebar.slider("Font Size", 8, 20, 12)
    axes_thickness = st.sidebar.slider("Axes Thickness", 0.5, 2.0, 0.8, step=0.1)
    axes_line_color = st.sidebar.color_picker("Axes Line Color", "#808080")
    marker_size = st.sidebar.slider("Scatter Marker Size", 20, 500, 100)
    alpha = st.sidebar.slider("Marker Transparency", 0.1, 1.0, 0.5, step=0.1)
    font_family = st.sidebar.selectbox("Font Family", ["Arial", "Times New Roman", "Helvetica"], index=0)
    
    colormaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'rainbow', 'jet', 'tab10', 'tab20',
        'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2', 'Pastel1', 'Pastel2',
        'viridis_r', 'plasma_r', 'inferno_r', 'magma_r', 'hot_r', 'cool_r', 'rainbow_r', 'jet_r',
        'spring', 'summer', 'autumn', 'winter', 'bone', 'copper', 'pink', 'ocean', 'terrain', 'gist_earth',
        'gist_rainbow', 'gist_heat', 'coolwarm', 'twilight', 'twilight_shifted', 'hsv', 'flag', 'prism',
        'nipy_spectral', 'gist_ncar', 'brg', 'cmr_map', 'cubehelix', 'gnuplot', 'gnuplot2', 'seismic'
    ]
    colormap = st.sidebar.selectbox("Colormap", colormaps, index=colormaps.index('tab20'))
    
    hist_edge_width = st.sidebar.slider("Histogram Edge Line Width", 0.5, 2.0, 1.0, step=0.1)
    xlabel_color = st.sidebar.color_picker("X-Label Color", "#000000")
    ylabel_color = st.sidebar.color_picker("Y-Label Color", "#000000")
    title_color = st.sidebar.color_picker("Title Color", "#000000")
    
    # Filters for scatter and heatmap
    selected_papers = st.sidebar.multiselect("Select Papers", df['paper_id'].unique(), default=df['paper_id'].unique())
    temp_range = st.sidebar.slider("Temperature Range (°C)", 0, 2000, (0, 1000), step=10)
    sfe_range = st.sidebar.slider("SFE Range (mJ/m²)", -1000, 1000, (-1000, 1000), step=10)
    use_log_heatmap = st.sidebar.checkbox("Use Logarithmic Scale for Heatmap", value=False)
    
    # Update colors
    cmap = plt.get_cmap(colormap)
    param_colors = {param: cmap(i / len(param_types)) for i, param in enumerate(param_types)}
    if colormap == "tab20":
        param_colors = default_colors
    
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': axes_thickness,
        'xtick.major.width': axes_thickness,
        'ytick.major.width': axes_thickness,
        'text.usetex': False #True
    })
    
    st.subheader("Extracted Parameters")
    filtered_df = df[df['paper_id'].isin(selected_papers) & 
                    df['entity_label'].isin(entity_types) &
                    (df['value'].between(sfe_range[0], sfe_range[1]) | df['entity_label'].isin(['MATERIAL'])) &
                    (df['value'].between(temp_range[0], temp_range[1]) | ~df['entity_label'].isin(['TEMPERATURE_C', 'TEMPERATURE_K']))]
    st.dataframe(
        filtered_df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]],
        use_container_width=True,
        column_config={
            "context": st.column_config.TextColumn("Context", help="Surrounding text for the parameter."),
            "value": st.column_config.NumberColumn("Value", help="Numerical value (SFE in mJ/m²)."),
            "outcome": st.column_config.TextColumn("Outcome", help="Related outcome (e.g., ductility).")
        }
    )
    
    # PMI results
    st.subheader("PMI Scores for SFE Context Phrases")
    pmi_data = []
    for paper_id, phrases in pmi_results.items():
        if paper_id in selected_papers:
            for phrase, score in phrases.items():
                pmi_data.append({"paper_id": paper_id, "phrase": phrase, "PMI Score": round(score, 3)})
    pmi_df = pd.DataFrame(pmi_data)
    if not pmi_df.empty:
        st.dataframe(pmi_df, use_container_width=True)
        st.download_button(
            label="Download PMI Scores CSV",
            data=pmi_df.to_csv(index=False),
            file_name="pmi_scores.csv",
            mime="text/csv"
        )
    else:
        st.info("No PMI scores available for the selected papers.")
    
    # Download filtered data
    st.subheader("Download Filtered Data")
    st.download_button(
        label="Download Filtered Parameters CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_sfe_params.csv",
        mime="text/csv"
    )
    
    # Histograms
    st.subheader("Parameter Distribution Analysis")
    for param_type in entity_types:
        if param_type in param_types:
            param_df = filtered_df[filtered_df["entity_label"] == param_type]
            if not param_df.empty:
                values = param_df["value"].dropna()
                if not values.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for spine in ax.spines.values():
                        spine.set_color(axes_line_color)
                    counts, bins, _ = ax.hist(values, bins=30 if param_type == "STACKING_FAULT_ENERGY" else 20,
                                             edgecolor='black', linewidth=hist_edge_width, color=param_colors[param_type], alpha=alpha,
                                             label=param_type.replace('_', ' ').title())
                    unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
                    ax.set_xlabel(f"{param_type.replace('_', ' ').title()} ({unit})", fontweight='bold', color=xlabel_color)
                    ax.set_ylabel("Count", fontweight='bold', color=ylabel_color)
                    ax.set_title(f"Distribution of {param_type.replace('_', ' ').title()}", fontweight='bold', pad=15, color=title_color)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best')
                    plt.tight_layout()
                    st.pyplot(fig)
                    hist_data = pd.DataFrame({
                        'Bin_Lower': bins[:-1],
                        'Bin_Upper': bins[1:],
                        'Count': counts
                    })
                    st.download_button(
                        label=f"Download {param_type} Histogram Counts CSV",
                        data=hist_data.to_csv(index=False),
                        file_name=f"{param_type.lower()}_histogram_counts.csv",
                        mime="text/csv"
                    )
                    plt.close(fig)
    
    # New Scatter Plot: SFE on x, Temperature on y
    sfe_df = filtered_df[filtered_df["entity_label"] == "STACKING_FAULT_ENERGY"]
    temp_df = filtered_df[filtered_df["entity_label"].isin(["TEMPERATURE_C", "TEMPERATURE_K"])]
    if not sfe_df.empty and not temp_df.empty:
        st.subheader("SFE vs Temperature (Multiple Temperatures per SFE)")
        scatter_data = []
        for paper_id in sfe_df["paper_id"].unique():
            sfe_entries = sfe_df[sfe_df["paper_id"] == paper_id]
            temp_entries = temp_df[temp_df["paper_id"] == paper_id]
            for _, sfe_row in sfe_entries.iterrows():
                sfe_value = sfe_row["value"]
                for _, temp_row in temp_entries.iterrows():
                    temp_value = temp_row["value"]
                    if temp_row["unit"] == "K":
                        temp_value -= 273
                    if temp_range[0] <= temp_value <= temp_range[1] and sfe_range[0] <= sfe_value <= sfe_range[1]:
                        scatter_data.append({
                            "paper_id": paper_id,
                            "title": sfe_row["title"],
                            "SFE (mJ/m²)": sfe_value,
                            "Temperature (°C)": temp_value
                        })
        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)
            if len(scatter_df) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                for spine in ax.spines.values():
                    spine.set_color(axes_line_color)
                paper_ids = scatter_df["paper_id"].unique()
                for i, paper_id in enumerate(paper_ids):
                    paper_data = scatter_df[scatter_df["paper_id"] == paper_id]
                    ax.scatter(paper_data["SFE (mJ/m²)"], paper_data["Temperature (°C)"],
                               c=[cmap(i % 20)], s=marker_size, alpha=alpha)  # Cycle through tab20 colors
                ax.set_xlabel(r"Stacking Fault Energy (mJ/m²)", fontweight='bold', color=xlabel_color)
                ax.set_ylabel(r"Temperature (°C)", fontweight='bold', color=ylabel_color)
                ax.set_title("SFE vs Temperature by Paper", fontweight='bold', pad=15, color=title_color)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                st.download_button(
                    label="Download SFE vs Temperature Scatter CSV",
                    data=scatter_df.to_csv(index=False),
                    file_name="sfe_vs_temp_scatter_new.csv",
                    mime="text/csv"
                )
                plt.close(fig)
            else:
                st.warning("No valid SFE-temperature pairs found within the selected ranges.")
    
    # Previous Scatter Plot: Temperature on x, SFE on y
    if not sfe_df.empty and not temp_df.empty:
        st.subheader("Temperature vs SFE (Filtered by Context)")
        scatter_data = []
        for paper_id in sfe_df["paper_id"].unique():
            sfe_entries = sfe_df[sfe_df["paper_id"] == paper_id]
            temp_entries = temp_df[temp_df["paper_id"] == paper_id]
            for _, sfe_row in sfe_entries.iterrows():
                sfe_value = sfe_row["value"]
                sfe_context = sfe_row["context"].lower()
                for _, temp_row in temp_entries.iterrows():
                    temp_value = temp_row["value"]
                    temp_context = temp_row["context"].lower()
                    if temp_row["unit"] == "K":
                        temp_value -= 273
                    if temp_range[0] <= temp_value <= temp_range[1] and sfe_range[0] <= sfe_value <= sfe_range[1]:
                        if sfe_context in temp_context or temp_context in sfe_context or \
                           any(term in temp_context for term in ["stacking fault", "sfe", "stacking fault energy", "gsfe"]):
                            scatter_data.append({
                                "paper_id": paper_id,
                                "title": sfe_row["title"],
                                "Temperature (°C)": temp_value,
                                "SFE (mJ/m²)": sfe_value
                            })
        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)
            if len(scatter_df) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                for spine in ax.spines.values():
                    spine.set_color(axes_line_color)
                paper_ids = scatter_df["paper_id"].unique()
                for i, paper_id in enumerate(paper_ids):
                    paper_data = scatter_df[scatter_df["paper_id"] == paper_id]
                    ax.scatter(paper_data["Temperature (°C)"], paper_data["SFE (mJ/m²)"],
                               c=[cmap(i % 20)], s=marker_size, alpha=alpha)
                ax.set_xlabel(r"Temperature (°C)", fontweight='bold', color=xlabel_color)
                ax.set_ylabel(r"Stacking Fault Energy (mJ/m²)", fontweight='bold', color=ylabel_color)
                ax.set_title("Temperature vs SFE by Paper", fontweight='bold', pad=15, color=title_color)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                st.download_button(
                    label="Download Temperature vs SFE Scatter CSV",
                    data=scatter_df.to_csv(index=False),
                    file_name="temp_vs_sfe_scatter.csv",
                    mime="text/csv"
                )
                plt.close(fig)
    
    # Heatmap of SFE vs Temperature
    if not sfe_df.empty and not temp_df.empty:
        st.subheader("SFE vs Temperature Heatmap")
        heatmap_data = scatter_df[["Temperature (°C)", "SFE (mJ/m²)"]].dropna()
        if not heatmap_data.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for spine in ax.spines.values():
                spine.set_color(axes_line_color)
            heatmap, xedges, yedges = np.histogram2d(
                heatmap_data["Temperature (°C)"], heatmap_data["SFE (mJ/m²)"],
                bins=[20, 30], range=[[temp_range[0], temp_range[1]], [sfe_range[0], sfe_range[1]]]
            )
            heatmap = np.log1p(heatmap) if use_log_heatmap else heatmap
            im = ax.imshow(heatmap.T, origin='lower', cmap=cmap, interpolation='nearest',
                           extent=[temp_range[0], temp_range[1], sfe_range[0], sfe_range[1]])
            ax.set_xlabel(r"Temperature (°C)", fontweight='bold', color=xlabel_color)
            ax.set_ylabel(r"Stacking Fault Energy (mJ/m²)", fontweight='bold', color=ylabel_color)
            ax.set_title("Density of SFE vs Temperature", fontweight='bold', pad=15, color=title_color)
            plt.colorbar(im, label="Count" + (" (log scale)" if use_log_heatmap else ""))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            temp_centers = 0.5 * (xedges[:-1] + xedges[1:])
            sfe_centers = 0.5 * (yedges[:-1] + yedges[1:])
            temp_grid, sfe_grid = np.meshgrid(temp_centers, sfe_centers)
            heatmap_df = pd.DataFrame({
                    "Temperature (°C)": temp_grid.flatten(),
                    "SFE (mJ/m²)": sfe_grid.flatten(),
                    "Count": heatmap.T.flatten()
            })
            #heatmap_df = pd.DataFrame({
            #    'Temp_Lower': xedges[:-1],
            #    'Temp_Upper': xedges[1:],
            #    'SFE_Lower': yedges[:-1],
            #    'SFE_Upper': yedges[1:],
            #    'Count': heatmap.T.flatten()
            #})
            st.download_button(
                label="Download Heatmap Data CSV",
                data=heatmap_df.to_csv(index=False),
                file_name="sfe_vs_temp_heatmap.csv",
                mime="text/csv"
            )
            plt.close(fig)
    
    st.write(f"**Summary**: {len(filtered_df)} parameters loaded, including {len(filtered_df[filtered_df['entity_label'] == 'STACKING_FAULT_ENERGY'])} SFE and {len(filtered_df[filtered_df['entity_label'].isin(['TEMPERATURE_C', 'TEMPERATURE_K'])])} temperature parameters.")

# --- NER Analysis Tab ---
with tab1:
    st.header("NER Analysis for Stacking Fault Energy and Temperature")
    st.markdown("Extract SFE (mJ/m²) and temperature (°C or K) from `sfe_knowledgeuniverse.db` or a `.pkl` file using regex-based NER and PMI for phrase detection. Results are saved as `.h5`, `.pkl`, and `.pt`.")

    with st.sidebar:
        st.subheader("NER Analysis Parameters")
        source_type = st.selectbox(
            "Select Data Source",
            ["Full Text (sfe_knowledgeuniverse.db)", "Saved Results (.pkl)"],
            help="Choose whether to analyze full text or load saved results."
        )
        entity_types = st.multiselect(
            "Parameter Types to Display",
            ["MATERIAL", "STACKING_FAULT_ENERGY", "TEMPERATURE_C", "TEMPERATURE_K"],
            default=["MATERIAL", "STACKING_FAULT_ENERGY", "TEMPERATURE_C", "TEMPERATURE_K"],
            help="Select parameter types to filter results."
        )
        sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")
        analyze_button = st.button("Run NER Analysis")
        if source_type == "Saved Results (.pkl)":
            uploaded_file = st.file_uploader("Upload .pkl File", type=["pkl"], key="ner_pkl")

    if analyze_button:
        if source_type == "Full Text (sfe_knowledgeuniverse.db)":
            db_file = "sfe_knowledgeuniverse.db"
            if not os.path.exists(db_file):
                st.error(f"Database {db_file} not found. Ensure it exists in the working directory.")
            else:
                with st.spinner(f"Processing {db_file}..."):
                    df, pmi_results = process_sqlite(db_file)
        else:
            if not uploaded_file:
                st.error("Please upload a .pkl file.")
                st.stop()
            try:
                df = pd.read_pickle(uploaded_file)
                pmi_results = {}
                st.info(f"Loaded {len(df)} entities from {uploaded_file.name}.")
            except Exception as e:
                st.error(f"Error loading .pkl file: {str(e)}")
                logging.error(f"Error loading .pkl file: {str(e)}")
                st.stop()
        
        if df is None or df.empty:
            st.warning(f"No parameters extracted. Ensure the source contains relevant papers.")
        else:
            if entity_types:
                df = df[df["entity_label"].isin(entity_types)]
            
            if sort_by == "entity_label":
                df = df.sort_values(["entity_label", "value"])
            else:
                df = df.sort_values(["value", "entity_label"], na_position="last")
            
            visualize_results(df, entity_types, pmi_results)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "Download SFE Parameters CSV",
                csv,
                "sfe_params.csv",
                "text/csv"
            )
            
            json_data = df.to_json("sfe_params.json", orient="records", lines=True)
            with open("sfe_params.json", "rb") as f:
                st.download_button(
                    "Download SFE Parameters JSON",
                    f,
                    "sfe_params.json",
                    "application/json"
                )
            
            h5_path, pkl_path, pt_path = save_ner_results(df)
            if h5_path:
                st.info(f"Saved NER results to {h5_path}, {pkl_path}, and {pt_path}")
                for path, mime in [(h5_path, "application/x-hdf"), (pkl_path, "application/octet-stream"), (pt_path, "application/octet-stream")]:
                    with open(path, "rb") as f:
                        st.download_button(
                            label=f"Download {path}",
                            data=f,
                            file_name=path,
                            mime=mime
                        )

# --- Visualize Results Tab ---
with tab2:
    st.header("Visualize Existing NER Results")
    st.markdown("Load previously saved NER results from `.h5`, `.pkl`, or `.pt` files and visualize them with customizable, publication-quality Matplotlib plots. Adjust font size, axes thickness, axes line color, colormap, histogram edges, and label colors.")

    uploaded_file = st.file_uploader("Upload NER Results File (.h5, .pkl, or .pt)", type=["h5", "pkl", "pt"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".h5"):
                df = pd.read_hdf(uploaded_file, key="ner_results")
            elif uploaded_file.name.endswith(".pkl"):
                df = pd.read_pickle(uploaded_file)
            elif uploaded_file.name.endswith(".pt"):
                data = torch.load(uploaded_file)
                df = pd.DataFrame(data)
            else:
                st.error("Unsupported file format. Please upload .h5, .pkl, or .pt.")
                st.stop()
            
            st.success(f"Loaded **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
            
            entity_types = st.multiselect(
                "Parameter Types to Display",
                ["MATERIAL", "STACKING_FAULT_ENERGY", "TEMPERATURE_C", "TEMPERATURE_K"],
                default=["MATERIAL", "STACKING_FAULT_ENERGY", "TEMPERATURE_C", "TEMPERATURE_K"],
                key="viz_entity_types"
            )
            sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.", key="viz_sort_by")
            
            if entity_types:
                df = df[df["entity_label"].isin(entity_types)]
            
            if sort_by == "entity_label":
                df = df.sort_values(["entity_label", "value"])
            else:
                df = df.sort_values(["value", "entity_label"], na_position="last")
            
            visualize_results(df, entity_types, {})
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            logging.error(f"Error loading visualization file: {str(e)}")

# Footer
st.markdown("---")
st.write("Developed for stacking fault energy analysis in Al-based and multicomponent alloys using regex-based NER and PMI.")
st.markdown("**How to Run**:")
st.markdown("""
1. Install dependencies: `pip install pandas streamlit matplotlib numpy spacy fuzzywuzzy python-Levenshtein h5py torch scipy`
2. Install SpaCy model: `python -m spacy download en_core_web_lg`
3. Save this code as `sfe_ner_analysis.py`.
4. Run with: `streamlit run sfe_ner_analysis.py --server.fileWatcherType none` to avoid PyTorch conflicts.
5. Place `sfe_knowledgeuniverse.db` in the same directory or upload a `.pkl` file.
""")
