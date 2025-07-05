import arxiv
import fitz  # PyMuPDF
import spacy
from spacy.matcher import Matcher
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import sqlite3
from collections import Counter
from datetime import datetime
import numpy as np
import logging
import time
from fuzzywuzzy import fuzz
import h5py
import pickle
import torch

# Initialize logging
logging.basicConfig(filename='sfe_analysis.log', level=logging.DEBUG)

# Initialize Streamlit app
st.set_page_config(page_title="SFE Analysis Tool for Al-based Alloys", layout="wide")
st.title("Stacking Fault Energy Analysis for Al-based and Multicomponent Alloys")
st.markdown("""
This tool supports research on stacking fault energy (SFE) in Al-based or multicomponent alloys. Use the **arXiv Query** tab to search for papers and download PDFs, storing metadata in `sfe_papers.db` and full text in `sfe_knowledgeuniverse.db`. The **NER Analysis** tab extracts SFE (J/m²) and temperature (°C or K) from either database. The **Visualize Results** tab loads and visualizes previously saved NER results from `.h5`, `.pkl`, or `.pt` files.
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `arxiv`, `pymupdf`, `spacy`, `pandas`, `streamlit`, `matplotlib`, `numpy`, `pyarrow`, `fuzzywuzzy`, `h5py`, `torch`
- Install with: `pip install arxiv pymupdf spacy pandas streamlit matplotlib numpy pyarrow fuzzywuzzy h5py torch`
- For optimal NER, install: `python -m spacy download en_core_web_lg`
""")

# Create PDFs directory
pdf_dir = "pdfs"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir} for storing PDFs.")

# Tabs for arXiv Query, NER Analysis, and Visualization
tab1, tab2, tab3 = st.tabs(["arXiv Query", "NER Analysis", "Visualize Results"])

# --- arXiv Query Tab ---
with tab1:
    st.header("arXiv Query for SFE in Al-based Alloys")
    st.markdown("Search arXiv for papers on stacking fault energy in Al-based or multicomponent alloys. Download PDFs and store metadata in `sfe_papers.db` and full text in `sfe_knowledgeuniverse.db`.")

    # Query arXiv function
    def query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases=[]):
        try:
            query_terms = query.strip().split()
            formatted_terms = []
            synonyms = {
                "stacking fault energy": ["SFE", "GSFE", "stacking faults", "SF", "stacking-fault energy", "stacking fault", "intrinsic stacking fault", "extrinsic stacking fault"],
                "alloys": ["Al alloys", "aluminum alloys", "alloy material", "AlSi", "multicomponent alloys", "AlMgSi alloys", "high-entropy alloys", "Al-based alloys", "HEA"],
                "temperature": ["Kelvin", "Celsius", "T", "thermal", "temp", "temperature-dependent"],
                "additive manufacturing": ["selective laser melting", "SLM", "3D printing", "metal additive manufacturing", "lpbf", "direct energy deposition", "laser powder bed fusion", "DED", "AM", "additively manufactured"],
            }
            api_terms = []
            for term in query_terms:
                if term.startswith('"') and term.endswith('"'):
                    api_terms.append(term.strip('"').replace(" ", "+"))
                else:
                    term = term.lower()
                    api_terms.append(term)
                    for key, syn_list in synonyms.items():
                        if term == key:
                            api_terms.extend(syn_list)
            api_query = " OR ".join(api_terms)
            for phrase in exact_phrases:
                api_query += f' AND "{phrase.replace(" ", "+")}"'
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=api_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            papers = []
            for result in client.results(search):
                if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                    abstract = result.summary.lower()
                    title = result.title.lower()
                    query_words = set(word.lower() for word in re.split(r'\s+|\".*?\"', query) if word and not word.startswith('"'))
                    for key, syn_list in synonyms.items():
                        if key in query_words:
                            query_words.update(syn_list)
                    matched_terms = []
                    for word in query_words:
                        if word in abstract or word in title:
                            matched_terms.append(word)
                        else:
                            for text in [abstract, title]:
                                words = text.split()
                                for w in words:
                                    if fuzz.partial_ratio(word, w) > 75:
                                        matched_terms.append(word)
                                        break
                    matched_terms = list(set(matched_terms))
                    if len(matched_terms) >= 1:
                        match_score = len(matched_terms) / max(1, len(query_words))
                        abstract_highlighted = abstract
                        for term in matched_terms:
                            abstract_highlighted = re.sub(r'\b{}\b'.format(term), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                        
                        papers.append({
                            "id": result.entry_id.split('/')[-1],
                            "title": result.title,
                            "year": result.published.year,
                            "categories": ", ".join(result.categories),
                            "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                            "abstract_highlighted": abstract_highlighted[:200] + "..." if len(abstract_highlighted) > 200 else abstract_highlighted,
                            "pdf_url": result.pdf_url,
                            "download_status": "Not downloaded",
                            "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                            "match_score": round(match_score * 100),
                            "pdf_path": None
                        })
                        logging.debug(f"Paper {result.entry_id} included with matched terms: {matched_terms}, score: {match_score}")
                    else:
                        logging.debug(f"Paper {result.entry_id} excluded: insufficient matched terms ({matched_terms})")
                if len(papers) >= max_results:
                    break
            
            if len(papers) < 10:
                logging.info(f"Fallback query triggered: too few papers ({len(papers)})")
                fallback_query = "stacking fault OR SFE OR aluminum alloys OR multicomponent alloys"
                search = arxiv.Search(
                    query=fallback_query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance,
                    sort_order=arxiv.SortOrder.Descending
                )
                for result in client.results(search):
                    if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                        abstract = result.summary.lower()
                        title = result.title.lower()
                        matched_terms = []
                        for word in ["sfe", "stacking fault", "al alloy", "multicomponent alloy"]:
                            if word in abstract or word in title:
                                matched_terms.append(word)
                            else:
                                for text in [abstract, title]:
                                    words = text.split()
                                    for w in words:
                                        if fuzz.partial_ratio(word, w) > 75:
                                            matched_terms.append(word)
                                            break
                        matched_terms = list(set(matched_terms))
                        if len(matched_terms) >= 1 and result.entry_id not in [p["id"] for p in papers]:
                            match_score = len(matched_terms) / 4
                            abstract_highlighted = abstract
                            for term in matched_terms:
                                abstract_highlighted = re.sub(r'\b{}\b'.format(term), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                            
                            papers.append({
                                "id": result.entry_id.split('/')[-1],
                                "title": result.title,
                                "year": result.published.year,
                                "categories": ", ".join(result.categories),
                                "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                                "abstract_highlighted": abstract_highlighted[:200] + "..." if len(abstract_highlighted) > 200 else abstract_highlighted,
                                "pdf_url": result.pdf_url,
                                "download_status": "Not downloaded",
                                "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                                "match_score": round(match_score * 100),
                                "pdf_path": None
                            })
                            logging.debug(f"Fallback: Paper {result.entry_id} included with matched terms: {matched_terms}, score: {match_score}")
                    if len(papers) >= max_results:
                        break
            
            logging.info(f"Total found {len(papers)} papers for query: {api_query}")
            print(f"Found {len(papers)} papers for query: {api_query}")
            return papers
        except Exception as e:
            logging.error(f"arXiv query failed: {str(e)}")
            st.error(f"Error querying arXiv: {str(e)}")
            return []

    # Download PDF function
    def download_pdf(pdf_url, paper_id):
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
        try:
            urllib.request.urlretrieve(pdf_url, pdf_path)
            file_size = os.path.getsize(pdf_path) / 1024  # Size in KB
            return f"Downloaded ({file_size:.2f} KB)", pdf_path
        except Exception as e:
            logging.error(f"PDF download failed for {paper_id}: {str(e)}")
            return f"Failed: {str(e)}", None

    # Extract text from PDF
    def extract_text_from_pdf(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logging.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
            return f"Error: {str(e)}"

    # Save to SQLite databases
    def save_to_sqlite(df, db_file="sfe_papers.db"):
        try:
            conn = sqlite3.connect(db_file)
            df.to_sql("papers", conn, if_exists="replace", index=False)
            conn.close()
            return f"Saved metadata to {db_file}"
        except Exception as e:
            logging.error(f"SQLite save failed for {db_file}: {str(e)}")
            return f"Failed to save to {db_file}: {str(e)}"

    def save_full_text_to_sqlite(papers, db_file="sfe_knowledgeuniverse.db"):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS full_text (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    year INTEGER,
                    full_text TEXT
                )
            """)
            for paper in papers:
                if paper["pdf_path"]:
                    text = extract_text_from_pdf(paper["pdf_path"])
                    if not text.startswith("Error"):
                        cursor.execute(
                            "INSERT OR REPLACE INTO full_text (paper_id, title, year, full_text) VALUES (?, ?, ?, ?)",
                            (paper["id"], paper["title"], paper["year"], text)
                        )
            conn.commit()
            conn.close()
            return f"Saved full text to {db_file}"
        except Exception as e:
            logging.error(f"Full text SQLite save failed for {db_file}: {str(e)}")
            return f"Failed to save full text to {db_file}: {str(e)}"

    def save_to_parquet(df, parquet_file="sfe_papers_metadata.parquet"):
        try:
            df.to_parquet(parquet_file, index=False)
            return f"Saved metadata to {parquet_file}"
        except Exception as e:
            logging.error(f"Parquet save failed: {str(e)}")
            return f"Failed to save to Parquet: {str(e)}"

    # Sidebar for search inputs
    with st.sidebar:
        st.subheader("arXiv Search Parameters")
        st.markdown("Customize your search for SFE in Al-based/multicomponent alloys.")
        
        query_option = st.radio(
            "Select Query Type",
            ["Default Query", "Custom Query", "Suggested Queries"],
            help="Choose how to specify the search query."
        )
        exact_phrases = []
        if query_option == "Default Query":
            query = "stacking fault energy Al alloys temperature"
            st.write("Using default query: **" + query + "**")
        elif query_option == "Custom Query":
            query = st.text_input("Enter Custom Query", value="stacking fault energy multicomponent alloys")
            exact_phrases_input = st.text_input("Exact Phrases (comma-separated, e.g., \"stacking faults\")", value="")
            exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
            st.write("Custom query: **" + query + "**")
            if exact_phrases:
                st.write("Exact phrases: **" + ", ".join(f'"{p}"' for p in exact_phrases) + "**")
        else:
            suggested_queries = [
                "stacking fault energy Al alloys",
                "stacking faults multicomponent alloys temperature",
                "stacking faults AlSi",
                "SFE AlMgSi alloys",
                "stacking fault aluminum alloys"
            ]
            query = st.selectbox("Choose Suggested Query", suggested_queries)
            exact_phrases_input = st.text_input("Exact Phrases (comma-separated, e.g., \"stacking faults\")", value="")
            exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
            st.write("Selected query: **" + query + "**")
            if exact_phrases:
                st.write("Exact phrases: **" + ", ".join(f'"{p}"' for p in exact_phrases) + "**")
        
        default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph", "cond-mat.mes-hall", "cond-mat.other"]
        extra_categories = ["physics.optics", "eess.sy", "cs.CE"]
        categories = st.multiselect(
            "Select arXiv Categories",
            default_categories + extra_categories,
            default=default_categories,
            help="Filter papers by categories."
        )
        
        max_results = st.slider(
            "Maximum Number of Papers",
            min_value=1,
            max_value=500,
            value=100,
            help="Set the maximum number of papers to retrieve."
        )
        
        current_year = datetime.now().year
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input(
                "Start Year",
                min_value=1900,
                max_value=current_year,
                value=1990,
                help="Earliest publication year."
            )
        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=start_year,
                max_value=current_year,
                value=current_year,
                help="Latest publication year."
            )
        
        output_formats = st.multiselect(
            "Select Output Formats",
            ["CSV", "SQLite (.db)", "Parquet (.parquet)", "JSON"],
            default=["CSV", "SQLite (.db)"],
            help="Choose formats for saving metadata."
        )
        
        search_button = st.button("Search arXiv")

    if search_button:
        if not query.strip():
            st.error("Please enter a valid query.")
        elif not categories:
            st.error("Please select at least one category.")
        elif start_year > end_year:
            st.error("Start year must be less than or equal to end year.")
        else:
            with st.spinner("Querying arXiv..."):
                papers = query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases)
            
            if not papers:
                st.warning("No papers found matching your criteria.")
                st.markdown("""
                **Suggestions to find more papers:**
                - Use broader terms (e.g., 'stacking faults aluminum').
                - Include more synonyms in exact phrases (e.g., "stacking faults").
                - Add more categories (e.g., 'cs.CE' for computational engineering).
                - Extend the year range (e.g., 1980–2025).
                - Increase the maximum number of papers (up to 500).
                - Check the log file (sfe_analysis.log) for details on excluded papers.
                """)
            else:
                st.success(f"Found **{len(papers)}** papers matching your query!")
                exact_display = ', '.join(f'"{p}"' for p in exact_phrases) if exact_phrases else 'None'
                st.write(f"Query: **{query}** | Exact Phrases: **{exact_display}**")
                st.write(f"Categories: **{', '.join(categories)}** | Years: **{start_year}–{end_year}**")
                
                st.subheader("Downloading PDFs")
                progress_bar = st.progress(0)
                for i, paper in enumerate(papers):
                    if paper["pdf_url"]:
                        status, pdf_path = download_pdf(paper["pdf_url"], paper["id"])
                        paper["download_status"] = status
                        paper["pdf_path"] = pdf_path
                    else:
                        paper["download_status"] = "No PDF URL"
                        paper["pdf_path"] = None
                    progress_bar.progress((i + 1) / len(papers))
                    time.sleep(0.1)
                
                df = pd.DataFrame(papers)
                st.subheader("Paper Details")
                st.dataframe(
                    df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "match_score", "download_status", "pdf_path"]],
                    use_container_width=True,
                    column_config={
                        "abstract_highlighted": st.column_config.TextColumn("Abstract (Highlighted)", help="Matched terms in bold orange.")
                    }
                )
                
                if "CSV" in output_formats:
                    csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                    csv_path = "sfe_papers_metadata.csv"
                    with open(csv_path, "w") as f:
                        f.write(csv)
                    st.info(f"Metadata CSV saved as {csv_path}. Automatic download starting...")
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            label="Download Paper Metadata CSV (Automatic)",
                            data=f,
                            file_name="sfe_papers_metadata.csv",
                            mime="text/csv",
                            key=f"auto_download_{time.time()}"
                        )
                    st.download_button(
                        label="Download Paper Metadata CSV (Manual)",
                        data=csv,
                        file_name="sfe_papers_metadata.csv",
                        mime="text/csv",
                        key="manual_download"
                    )
                
                if "SQLite (.db)" in output_formats:
                    sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), "sfe_papers.db")
                    st.info(sqlite_status)
                    full_text_status = save_full_text_to_sqlite(papers, "sfe_knowledgeuniverse.db")
                    st.info(full_text_status)
                
                if "Parquet (.parquet)" in output_formats:
                    parquet_status = save_to_parquet(df.drop(columns=["abstract_highlighted"]), "sfe_papers_metadata.parquet")
                    st.info(parquet_status)
                
                if "JSON" in output_formats:
                    json_path = "sfe_papers_metadata.json"
                    df.drop(columns=["abstract_highlighted"]).to_json(json_path, orient="records", lines=True)
                    st.info(f"Saved metadata to {json_path}")
                    with open(json_path, "rb") as f:
                        st.download_button(
                            label="Download Paper Metadata JSON",
                            data=f,
                            file_name="sfe_papers_metadata.json",
                            mime="application/json",
                            key="json_download"
                        )
                
                downloaded = sum(1 for p in papers if "Downloaded" in p["download_status"])
                st.write(f"**Summary**: {len(papers)} papers found, {downloaded} PDFs downloaded successfully.")
                if downloaded < len(papers):
                    st.warning("Some PDFs failed to download. Check 'download_status' for details.")
                common_terms = set()
                for terms in df["matched_terms"]:
                    if terms and terms != "None":
                        common_terms.update(terms.split(", "))
                if common_terms:
                    st.markdown(f"**Query Refinement Tip**: Common matched terms: {', '.join(common_terms)}. Try focusing on these (e.g., '{' '.join(list(common_terms)[:3])}').")

# --- NER Analysis Tab ---
with tab2:
    st.header("NER Analysis for Stacking Fault Energy and Temperature")
    st.markdown("Extract stacking fault energy (SFE in J/m²) and temperature (°C or K) from either `sfe_papers.db` (metadata) or `sfe_knowledgeuniverse.db` (full text). Results are saved as `.h5`, `.pkl`, and `.pt` for reuse.")

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_lg")
    except Exception as e:
        st.warning(f"Failed to load 'en_core_web_lg': {e}. Falling back to 'en_core_web_sm'.")
        try:
            nlp = spacy.load("en_core_web_sm")
            st.info("Using en_core_web_sm (less accurate). Install en_core_web_lg: `python -m spacy download en_core_web_lg`")
        except Exception as e2:
            st.error(f"Failed to load spaCy model: {e2}. Install with: `python -m spacy download en_core_web_sm`")
            st.stop()

    # Custom NER patterns
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?([eE][-+]?\d+)?$"}}, {"LOWER": {"IN": ["j/m²", "j/m^2", "j m^-2", "mj/m²", "mj/m^2", "mj m^-2"]}}, {"LOWER": {"IN": ["stacking faults", "sfe", "sf", "stacking fault energy", "gsfe"]}, "OP": "?" }],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?([eE][-+]?\d+)?$"}}, {"LOWER": {"IN": ["°c", "celsius"]}}],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?([eE][-+]?\d+)?$"}}, {"LOWER": {"IN": ["k", "kelvin"]}}]
    ]
    param_types = ["STACKING_FAULT_ENERGY", "TEMPERATURE_C", "TEMPERATURE_K"]
    for i, pattern in enumerate(patterns):
        matcher.add(f"ALLOY_PARAM_{param_types[i]}", [pattern])

    # Parameter validation ranges (in SI units for SFE)
    valid_ranges = {
        "STACKING_FAULT_ENERGY": (0.000001, 1, "J/m²"),
        "TEMPERATURE_C": (0, 1000, "°C"),
        "TEMPERATURE_K": (273, 1273, "K")
    }

    # Color map for parameter histograms
    param_colors = {param: cm.tab10(i / len(param_types)) for i, param in enumerate(param_types)}

    # Perform NER
    def extract_alloy_parameters(text):
        try:
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ["MATERIAL", "ORG", "PRODUCT"] or any(term in ent.text.lower() for term in ["al alloy", "alsi", "almgsi", "multicomponent alloy", "aluminum alloy", "hea"]):
                    entities.append({
                        "text": ent.text,
                        "label": "MATERIAL",
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "value": None,
                        "unit": None,
                        "outcome": None
                    })
            
            matches = matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                label = nlp.vocab.strings[match_id].replace("ALLOY_PARAM_", "")
                match_text = span.text
                value_match = re.match(r"(\d+\.?\d*)([eE][-+]?\d+)?", match_text)
                value = float(value_match.group(0)) if value_match else None
                unit = match_text[value_match.end():].strip() if value_match else None
                
                if label == "STACKING_FAULT_ENERGY" and value is not None:
                    if unit.lower() in ["mj/m²", "mj/m^2", "mj m^-2"]:
                        value /= 1000  # Convert mJ/m² to J/m²
                        unit = "J/m²"
                    context_start = max(0, start - 100)
                    context_end = min(len(text), end + 100)
                    context_text = text[context_start:context_end].lower()
                    sfe_terms = ["stacking faults", "sfe", "sf", "stacking fault energy", "gsfe"]
                    if not any(term in context_text for term in sfe_terms):
                        for token in doc[start:end]:
                            if any(fuzz.partial_ratio(token.text.lower(), term) > 75 for term in sfe_terms):
                                break
                        else:
                            logging.debug(f"Skipping entity at {start}-{end}: no stacking fault context ({match_text})")
                            continue
                
                if label == "TEMPERATURE_K" and value is not None:
                    if 0 <= value <= 100:
                        label = "TEMPERATURE_C"
                        unit = "°C"
                
                if label in valid_ranges and value is not None:
                    min_val, max_val, expected_unit = valid_ranges[label]
                    if not (min_val <= value <= max_val and (unit == expected_unit or unit is None)):
                        logging.debug(f"Skipping entity {match_text}: value {value} outside range {min_val}-{max_val} or unit {unit} != {expected_unit}")
                        continue
                
                outcome = None
                context_start = max(0, start - 100)
                context_end = min(len(text), end + 100)
                context_text = text[context_start:context_end].lower()
                outcome_terms = ["ductility", "strength", "hardness", "deformation", "twinning"]
                for term in outcome_terms:
                    if term in context_text:
                        outcome = term
                        break
                
                entities.append({
                    "text": span.text,
                    "label": label,
                    "start": start,
                    "end": end,
                    "value": value,
                    "unit": unit,
                    "outcome": outcome
                })
                logging.debug(f"Extracted entity: {span.text}, label: {label}, value: {value}, unit: {unit}")
            
            return entities
        except Exception as e:
            logging.error(f"NER failed: {str(e)}")
            return [{"text": f"Error: {str(e)}", "label": "ERROR", "start": 0, "end": 0, "value": None, "unit": None, "outcome": None}]

    # Save NER results to HDF5, Pickle, and PyTorch
    def save_ner_results(df, base_name="sfe_params"):
        try:
            # HDF5
            h5_path = f"{base_name}.h5"
            df.to_hdf(h5_path, key="ner_results", mode="w")
            # Pickle
            pkl_path = f"{base_name}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(df, f)
            # PyTorch
            pt_path = f"{base_name}.pt"
            torch.save(df.to_dict(orient="records"), pt_path)
            return h5_path, pkl_path, pt_path
        except Exception as e:
            logging.error(f"Failed to save NER results: {str(e)}")
            return None, None, None

    # Process SQLite database for NER
    def process_sqlite(db_file, source_type):
        try:
            conn = sqlite3.connect(db_file)
            if source_type == "Metadata":
                df = pd.read_sql("SELECT * FROM papers WHERE abstract IS NOT NULL", conn)
                results = []
                relevant_entries = 0
                for _, row in df.iterrows():
                    abstract = row["abstract"].lower()
                    if not any(term in abstract for term in ["stacking fault energy", "sfe", "stacking fault", "sf", "gsfe", "al alloy", "multicomponent alloy", "aluminum alloy"]):
                        logging.debug(f"Skipping paper {row['id']}: no relevant terms in abstract")
                        continue
                    relevant_entries += 1
                    entities = extract_alloy_parameters(abstract)
                    for entity in entities:
                        results.append({
                            "paper_id": row["id"],
                            "title": row["title"],
                            "year": row["year"],
                            "entity_text": entity["text"],
                            "entity_label": entity["label"],
                            "value": entity["value"],
                            "unit": entity["unit"],
                            "outcome": entity["outcome"],
                            "context": abstract[max(0, entity["start"] - 50):min(len(abstract), entity["end"] + 50)].replace("\n", " ")
                        })
                st.info(f"Processed {relevant_entries} relevant metadata entries from {db_file}.")
            else:  # Full Text
                df = pd.read_sql("SELECT * FROM full_text WHERE full_text IS NOT NULL", conn)
                results = []
                relevant_entries = 0
                for _, row in df.iterrows():
                    text = row["full_text"].lower()
                    if not any(term in text for term in ["stacking fault energy", "sfe", "stacking fault", "sf", "gsfe", "al alloy", "multicomponent alloy", "aluminum alloy"]):
                        logging.debug(f"Skipping paper {row['paper_id']}: no relevant terms in full text")
                        continue
                    relevant_entries += 1
                    entities = extract_alloy_parameters(text)
                    for entity in entities:
                        results.append({
                            "paper_id": row["paper_id"],
                            "title": row["title"],
                            "year": row["year"],
                            "entity_text": entity["text"],
                            "entity_label": entity["label"],
                            "value": entity["value"],
                            "unit": entity["unit"],
                            "outcome": entity["outcome"],
                            "context": text[max(0, entity["start"] - 50):min(len(text), entity["end"] + 50)].replace("\n", " ")
                        })
                st.info(f"Processed {relevant_entries} relevant full-text entries from {db_file}.")
            conn.close()
            return pd.DataFrame(results)
        except Exception as e:
            st.error(f"Error processing {db_file}: {str(e)}")
            logging.error(f"Error processing {db_file}: {str(e)}")
            return None

    # Sidebar for NER inputs
    with st.sidebar:
        st.subheader("NER Analysis Parameters")
        st.markdown("Configure the analysis to extract SFE and temperature.")
        
        source_type = st.selectbox(
            "Select Data Source",
            ["Metadata (sfe_papers.db)", "Full Text (sfe_knowledgeuniverse.db)"],
            help="Choose whether to analyze metadata or full text."
        )
        db_file = "sfe_papers.db" if source_type == "Metadata (sfe_papers.db)" else "sfe_knowledgeuniverse.db"
        entity_types = st.multiselect(
            "Parameter Types to Display",
            ["MATERIAL", "STACKING_FAULT_ENERGY", "TEMPERATURE_C", "TEMPERATURE_K"],
            default=["MATERIAL", "STACKING_FAULT_ENERGY", "TEMPERATURE_C"],
            help="Select parameter types to filter results."
        )
        sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")
        analyze_button = st.button("Run NER Analysis")

    if analyze_button:
        if not os.path.exists(db_file):
            st.error(f"Database {db_file} not found. Run the arXiv query first to generate it.")
        else:
            with st.spinner(f"Processing {db_file}..."):
                df = process_sqlite(db_file, source_type.split()[0])
            
            if df is None or df.empty:
                st.warning(f"No parameters extracted from {db_file}. Ensure the database contains relevant papers.")
            else:
                st.success(f"Extracted **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
                
                if entity_types:
                    df = df[df["entity_label"].isin(entity_types)]
                
                if sort_by == "entity_label":
                    df = df.sort_values(["entity_label", "value"])
                else:
                    df = df.sort_values(["value", "entity_label"], na_position="last")
                
                st.subheader("Extracted Parameters")
                st.dataframe(
                    df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]],
                    use_container_width=True,
                    column_config={
                        "context": st.column_config.TextColumn("Context", help="Surrounding text for the parameter."),
                        "value": st.column_config.NumberColumn("Value", help="Numerical value (SFE in J/m²)."),
                        "outcome": st.column_config.TextColumn("Outcome", help="Related outcome (e.g., ductility).")
                    }
                )
                
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
                
                st.subheader("SFE Distribution Analysis")
                for param_type in entity_types:
                    if param_type in param_types:
                        param_df = df[df["entity_label"] == param_type]
                        if not param_df.empty:
                            values = param_df["value"].dropna()
                            if not values.empty:
                                fig, ax = plt.subplots()
                                ax.hist(values, bins=10, edgecolor="black", color=param_colors[param_type])
                                unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
                                ax.set_xlabel(f"{param_type} ({unit})")
                                ax.set_ylabel("Count")
                                ax.set_title(f"Distribution of {param_type}")
                                st.pyplot(fig)
                
                sfe_df = df[df["entity_label"] == "STACKING_FAULT_ENERGY"]
                temp_df = df[df["entity_label"].isin(["TEMPERATURE_C", "TEMPERATURE_K"])]
                if not sfe_df.empty and not temp_df.empty:
                    fig, ax = plt.subplots()
                    for paper_id in sfe_df["paper_id"].unique():
                        sfe_values = sfe_df[sfe_df["paper_id"] == paper_id]["value"].dropna()
                        temp_values = temp_df[temp_df["paper_id"] == paper_id]["value"].dropna()
                        temp_units = temp_df[temp_df["paper_id"] == paper_id]["unit"].dropna()
                        if not sfe_values.empty and not temp_values.empty:
                            for temp, unit in zip(temp_values, temp_units):
                                if unit == "K":
                                    temp -= 273
                                for sfe in sfe_values:
                                    ax.scatter(temp, sfe, c=param_colors["STACKING_FAULT_ENERGY"], alpha=0.5)
                    ax.set_xlabel("Temperature (°C)")
                    ax.set_ylabel("Stacking Fault Energy (J/m²)")
                    ax.set_title("SFE vs Temperature")
                    st.pyplot(fig)
                
                st.write(f"**Summary**: {len(df)} parameters extracted, including {len(df[df['entity_label'] == 'STACKING_FAULT_ENERGY'])} SFE and {len(df[df['entity_label'].isin(['TEMPERATURE_C', 'TEMPERATURE_K'])])} temperature parameters.")
                st.markdown("""
                **Next Steps**:
                - Filter by parameter types to focus on SFE or temperature.
                - Review outcomes to link parameters to material properties.
                - Use saved `.h5`, `.pkl`, or `.pt` files in the Visualize Results tab.
                - Check sfe_analysis.log for debugging information if SFE extraction is low.
                """)

# --- Visualize Results Tab ---
with tab3:
    st.header("Visualize Existing NER Results")
    st.markdown("Load previously saved NER results from `.h5`, `.pkl`, or `.pt` files and visualize them as histograms and scatter plots.")

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
                default=["MATERIAL", "STACKING_FAULT_ENERGY", "TEMPERATURE_C"],
                help="Select parameter types to filter results.",
                key="viz_entity_types"
            )
            sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.", key="viz_sort_by")
            
            if entity_types:
                df = df[df["entity_label"].isin(entity_types)]
            
            if sort_by == "entity_label":
                df = df.sort_values(["entity_label", "value"])
            else:
                df = df.sort_values(["value", "entity_label"], na_position="last")
            
            st.subheader("Extracted Parameters")
            st.dataframe(
                df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]],
                use_container_width=True,
                column_config={
                    "context": st.column_config.TextColumn("Context", help="Surrounding text for the parameter."),
                    "value": st.column_config.NumberColumn("Value", help="Numerical value (SFE in J/m²)."),
                    "outcome": st.column_config.TextColumn("Outcome", help="Related outcome (e.g., ductility).")
                }
            )
            
            st.subheader("SFE Distribution Analysis")
            for param_type in entity_types:
                if param_type in param_types:
                    param_df = df[df["entity_label"] == param_type]
                    if not param_df.empty:
                        values = param_df["value"].dropna()
                        if not values.empty:
                            fig, ax = plt.subplots()
                            ax.hist(values, bins=10, edgecolor="black", color=param_colors[param_type])
                            unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
                            ax.set_xlabel(f"{param_type} ({unit})")
                            ax.set_ylabel("Count")
                            ax.set_title(f"Distribution of {param_type}")
                            st.pyplot(fig)
            
            sfe_df = df[df["entity_label"] == "STACKING_FAULT_ENERGY"]
            temp_df = df[df["entity_label"].isin(["TEMPERATURE_C", "TEMPERATURE_K"])]
            if not sfe_df.empty and not temp_df.empty:
                fig, ax = plt.subplots()
                for paper_id in sfe_df["paper_id"].unique():
                    sfe_values = sfe_df[sfe_df["paper_id"] == paper_id]["value"].dropna()
                    temp_values = temp_df[temp_df["paper_id"] == paper_id]["value"].dropna()
                    temp_units = temp_df[temp_df["paper_id"] == paper_id]["unit"].dropna()
                    if not sfe_values.empty and not temp_values.empty:
                        for temp, unit in zip(temp_values, temp_units):
                            if unit == "K":
                                temp -= 273
                            for sfe in sfe_values:
                                ax.scatter(temp, sfe, c=param_colors["STACKING_FAULT_ENERGY"], alpha=0.5)
                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("Stacking Fault Energy (J/m²)")
                ax.set_title("SFE vs Temperature")
                st.pyplot(fig)
            
            st.write(f"**Summary**: {len(df)} parameters loaded, including {len(df[df['entity_label'] == 'STACKING_FAULT_ENERGY'])} SFE and {len(df[df['entity_label'].isin(['TEMPERATURE_C', 'TEMPERATURE_K'])])} temperature parameters.")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            logging.error(f"Error loading visualization file: {str(e)}")

# Footer
st.markdown("---")
st.write("Developed for stacking fault energy analysis in Al-based and multicomponent alloys.")
