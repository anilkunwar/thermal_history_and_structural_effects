import streamlit as st
import PyPDF2
import tempfile
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import logging
import seaborn as sns
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
import spacy
from math import log
import uuid
import json
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK and spaCy data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK data already present.")
    except LookupError:
        try:
            logger.info("Downloading NLTK punkt_tab and stopwords...")
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            logger.info("NLTK data downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {str(e)}")
            st.error(f"Failed to download NLTK data: {str(e)}. Please try again or check your network.")
            return False
    return True

# Download spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy en_core_web_sm model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK data at startup
if not download_nltk_data():
    st.stop()

# Load IDF_APPROX from JSON or use hardcoded fallback
try:
    json_path = os.path.join(os.path.dirname(__file__), "idf_approx.json")
    with open(json_path, "r") as f:
        IDF_APPROX = json.load(f)
    logger.info("Loaded arXiv-derived IDF_APPROX from idf_approx.json")
except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
    logger.warning(f"Failed to load idf_approx.json from {json_path}: {str(e)}. Using default IDF_APPROX.")
    st.warning(f"Could not load idf_approx.json: {str(e)}. Falling back to hardcoded IDF values. Please ensure idf_approx.json is in the same directory as the script in the GitHub repository.")
    IDF_APPROX = {
        "study": log(1000 / 800), "analysis": log(1000 / 700), "results": log(1000 / 600),
        "method": log(1000 / 500), "experiment": log(1000 / 400),
        "spectroscopy": log(1000 / 50), "nanoparticle": log(1000 / 40), "diffraction": log(1000 / 30),
        "microscopy": log(1000 / 20), "quantum": log(1000 / 10),
        "selective laser melting": log(1000 / 50), "bimodal microstructure": log(1000 / 5),
        "stacking faults": log(1000 / 5), "al-si-mg": log(1000 / 10), "strength-ductility": log(1000 / 5),
        "finite element analysis": log(1000 / 25), "molecular dynamics": log(1000 / 20),
        "scanning electron microscopy": log(1000 / 15), "transmission electron microscopy": log(1000 / 15),
        "slm-fabricated alsimg1.4zr": log(1000 / 10), "grain refinement": log(1000 / 5),
        "dislocation dynamics": log(1000 / 5), "heterogeneous nucleation": log(1000 / 5),
        "melt pool dynamics": log(1000 / 5), "thermal gradient": log(1000 / 5)
    }
DEFAULT_IDF = log(100000 / 10000)  # Match arXiv corpus size (N=100,000)

# Define keyword categories
KEYWORD_CATEGORIES = {
    "Materials": [
        "alloy", "polymer", "nanoparticle", "crystal", "metal", "ceramic", "composite", "semiconductor",
        "graphene", "nanotube", "oxide", "thin film", "superconductor", "biomaterial", "aluminum", "magnesium",
        "zirconium", "silicon", "intermetallic", "al3zr",
        "aluminum alloy", "al-si-mg", "alsimg1.4zr", "lightweight alloy", "high-mg-content", "zr-modified alloy",
        "slm-fabricated alsimg1.4zr"
    ],
    "Methods": [
        "microscopy", "lithography", "deposition", "etching", "annealing", "characterization", "synthesis",
        "fabrication", "imaging", "scanning", "tomography", "spectroscopy", "diffraction", "ablation",
        "crystallization", "polymerization", "evaporation", "sputtering", "measurement", "aging",
        "electropolishing", "ion-beam thinning", "tensile testing",
        "selective laser melting", "vacuum induction", "gas atomization", "finite element analysis",
        "molecular dynamics", "electron backscatter diffraction", "energy-dispersive x-ray", "x-ray diffraction",
        "scanning electron microscopy", "transmission electron microscopy", "direct aging", "bidirectional scanning",
        "laser processing", "thermomechanical treatment", "severe plastic deformation"
    ],
    "Physical Phenomena": [
        "diffusion", "scattering", "conductivity", "magnetism", "superconductivity", "fluorescence", "polarization",
        "refraction", "absorption", "emission", "quantum", "thermal", "solidification", "deformation", "nucleation",
        "dislocation", "twinning", "stacking fault",
        "thermal gradient", "stacking faults", "dislocation dynamics", "grain refinement", "heterogeneous nucleation",
        "bimodal microstructure", "melt pool dynamics", "rapid cooling", "thermal history"
    ],
    "Properties": [
        "hardness", "conductivity", "resistivity", "magnetization", "density", "strength", "elasticity", "viscosity",
        "porosity", "permeability", "ductility", "toughness", "plasticity", "elongation", "yield strength",
        "ultimate tensile strength", "work hardening",
        "strength-ductility", "grain size", "thermal expansion", "stacking fault energy", "mechanical properties",
        "bimodal grains", "hall-petch", "work hardening rate", "strain hardening"
    ],
    "Other": [
        "sustainability", "industry 4.0", "optimization", "simulation", "modeling", "energy efficiency",
        "additive manufacturing", "sustainable manufacturing", "process-structure", "hierarchical microstructure",
        "non-uniform temperature", "laser-material interaction", "melt boundary"
    ]
}

# Define physics-related categories for attention mechanism
PHYSICS_CATEGORIES = ["Physical Phenomena", "Properties"]

# Define available colormaps for visualizations
#COLORMAPS = [
#    "viridis", "plasma", "inferno", "magma", "hot",
#    "coolwarm", "RdBu", "PiYG", "tab10", "Set2"
#]
COLORMAPS = [
    # Perceptually Uniform Sequential
    "viridis", "plasma", "inferno", "magma", "cividis",

    # Sequential
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu",
    "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",

    # Sequential (2)
    "cubehelix", "binary", "gist_yarg", "gist_gray", "gray", "bone",
    "pink", "spring", "summer", "autumn", "winter",

    # Diverging
    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn",
    "Spectral", "coolwarm", "bwr", "seismic",

    # Cyclic
    "twilight", "twilight_shifted", "hsv",

    # Qualitative
    "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3",
    "tab10", "tab20", "tab20b", "tab20c",

    # Miscellaneous
    "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern", "gnuplot",
    "gnuplot2", "CMRmap", "cubehelix", "brg", "gist_rainbow", "rainbow",
    "jet", "nipy_spectral", "gist_ncar",

    # Reversed versions
    "viridis_r", "plasma_r", "inferno_r", "magma_r", "cividis_r", "Greys_r",
    "Purples_r", "Blues_r", "Greens_r", "Oranges_r", "Reds_r", "YlOrBr_r",
    "YlOrRd_r", "OrRd_r", "PuRd_r", "RdPu_r", "BuPu_r", "GnBu_r", "PuBu_r",
    "YlGnBu_r", "PuBuGn_r", "BuGn_r", "YlGn_r", "twilight_r", "twilight_shifted_r",
    "hsv_r", "Spectral_r", "coolwarm_r", "bwr_r", "seismic_r", "RdBu_r",
    "PiYG_r", "PRGn_r", "BrBG_r", "PuOr_r", "RdGy_r", "RdYlBu_r", "RdYlGn_r",

    # Optional: If using `colorcet` or `cmocean`
    # (uncomment if these libraries are installed)
    # "cet_fire", "cet_ice", "cet_rainbow", "cet_colorwheel",
    # "cmo.haline", "cmo.thermal", "cmo.solar", "cmo.ice", "cmo.gray",
    # "cmo.matter", "cmo.turbid", "cmo.speed", "cmo.deep", "cmo.delta",
    # "cmo.amp", "cmo.phase", "cmo.balance", "cmo.diff", "cmo.curl", "cmo.tarn"
]

# Define available network styles
NETWORK_STYLES = [
    "seaborn-v0_8-white", "default", "ggplot", "bmh", "classic"
]

# Define node, edge, and label style options
NODE_SHAPES = ['o', 's', '^', 'v', '>', '<', 'd']  # Circle, square, triangles, diamond
EDGE_STYLES = ['solid', 'dashed', 'dotted', 'dashdot']
COLORS = ['black', 'red', 'blue', 'green', 'white']
FONT_FAMILIES = ['Arial', 'Helvetica', 'Times New Roman', 'Courier New']
BBOX_COLORS = ['black', 'white', 'gray', 'lightgray']
LAYOUT_ALGORITHMS = [
    'spring', 'circular', 'kamada_kawai', 'shell',
    'spectral', 'random', 'spiral', 'planar'
]

# Function to estimate IDF for terms not in IDF_APPROX
def estimate_idf(term, word_freq, total_words, idf_approx, keyword_categories, nlp_model):
    """
    Estimate IDF for a term/phrase not in IDF_APPROX using frequency, similarity, and category heuristics.
    Args:
        term (str): Term or phrase to estimate IDF for.
        word_freq (Counter): Frequency of terms in the PDF.
        total_words (int): Total words in the PDF.
        idf_approx (dict): Precomputed IDF values.
        keyword_categories (dict): Categories with associated keywords.
        nlp_model: spaCy model for similarity computation.
    Returns:
        float: Estimated IDF value.
    """
    # Initialize session state for caching IDF estimates
    if 'custom_idf' not in st.session_state:
        st.session_state.custom_idf = {}

    # Check cache
    if term in st.session_state.custom_idf:
        logger.debug(f"Using cached IDF for {term}: {st.session_state.custom_idf[term]}")
        return st.session_state.custom_idf[term]

    # Frequency-based heuristic: Lower TF suggests higher IDF
    tf = word_freq.get(term, 1) / total_words
    freq_idf = log(1 / max(tf, 1e-6))  # Avoid division by zero
    freq_idf = min(freq_idf, 8.517)  # Cap at max IDF_APPROX value (e.g., slm-fabricated alsimg1.4zr)

    # Similarity-based heuristic: Find closest IDF_APPROX term
    sim_idf = DEFAULT_IDF
    max_similarity = 0.0
    term_doc = nlp_model(term)
    for known_term in idf_approx:
        known_doc = nlp_model(known_term)
        similarity = term_doc.similarity(known_doc)
        if similarity > max_similarity and similarity > 0.7:  # Threshold for meaningful similarity
            max_similarity = similarity
            sim_idf = idf_approx[known_term]
            logger.debug(f"Similarity match for {term}: {known_term} (sim={similarity:.2f}, IDF={sim_idf:.3f})")

    # Category-based heuristic: Use average IDF of category if term matches
    cat_idf = DEFAULT_IDF
    for category, keywords in keyword_categories.items():
        if any(k in term or term in k for k in keywords):
            cat_idfs = [idf_approx.get(k, DEFAULT_IDF) for k in keywords if k in idf_approx]
            if cat_idfs:
                cat_idf = sum(cat_idfs) / len(cat_idfs)
                logger.debug(f"Category match for {term}: {category} (avg IDF={cat_idf:.3f})")
                break

    # Combine heuristics: Weighted average (favor similarity if strong match)
    if max_similarity > 0.7:
        estimated_idf = 0.7 * sim_idf + 0.2 * freq_idf + 0.1 * cat_idf
    else:
        estimated_idf = 0.4 * freq_idf + 0.4 * cat_idf + 0.2 * DEFAULT_IDF

    # Ensure IDF is within reasonable bounds
    estimated_idf = max(2.303, min(8.517, estimated_idf))  # Match IDF_APPROX range
    st.session_state.custom_idf[term] = estimated_idf
    logger.debug(f"Estimated IDF for {term}: {estimated_idf:.3f} (freq={freq_idf:.3f}, sim={sim_idf:.3f}, cat={cat_idf:.3f})")
    return estimated_idf

# Modified get_candidate_keywords to include frequency in idf_sources
def get_candidate_keywords(text, min_freq, min_length, use_stopwords, custom_stopwords, exclude_keywords, top_limit, tfidf_weight, use_nouns_only, include_phrases):
    stop_words = set(stopwords.words('english')) if use_stopwords else set()
    stop_words.update(['introduction', 'conclusion', 'section', 'chapter', 'the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
    stop_words.update([w.strip().lower() for w in custom_stopwords.split(",") if w.strip()])
    exclude_set = set([w.strip().lower() for w in exclude_keywords.split(",") if w.strip()])

    # Extract single words
    words = word_tokenize(text.lower())
    if use_nouns_only:
        doc = nlp(text)
        nouns = {token.text.lower() for token in doc if token.pos_ == "NOUN"}
        filtered_words = [w for w in words if w in nouns and w.isalnum() and len(w) >= min_length and w not in stop_words and w not in exclude_set]
    else:
        filtered_words = [w for w in words if w.isalnum() and len(w) >= min_length and w not in stop_words and w not in exclude_set]
    
    word_freq = Counter(filtered_words)
    logger.debug("Single word frequencies: %s", word_freq.most_common(20))

    # Extract phrases if enabled
    phrases = []
    if include_phrases:
        doc = nlp(text)
        raw_phrases = [
            chunk.text.lower() for chunk in doc.noun_chunks 
            if len(chunk.text.split()) > 1 and len(chunk.text) >= min_length
        ]
        # Clean phrases by removing leading/trailing stopwords
        phrases = [clean_phrase(phrase, stop_words) for phrase in raw_phrases if clean_phrase(phrase, stop_words)]
        phrases = [p for p in phrases if p not in stop_words and p not in exclude_set]
        phrase_freq = Counter(phrases)
        phrases = [(p, f) for p, f in phrase_freq.items() if f >= min_freq]
        logger.debug("Extracted phrases: %s", phrases[:20])

    # Compute TF-IDF scores with attention mechanism and track IDF sources
    total_words = len(word_tokenize(text))
    tfidf_scores = {}
    idf_sources = {}  # Track IDF source, value, and frequency for each term
    for word, freq in word_freq.items():
        if freq < min_freq:
            continue
        tf = freq / total_words
        if word in IDF_APPROX:
            idf = IDF_APPROX[word]
            source = "JSON"
        else:
            idf = estimate_idf(word, word_freq, total_words, IDF_APPROX, KEYWORD_CATEGORIES, nlp)
            source = "Estimated"
        tfidf_scores[word] = tf * idf * tfidf_weight
        idf_sources[word] = {"idf": idf, "source": source, "frequency": freq}
        logger.debug(f"PDF term {word}: TF-IDF={tfidf_scores[word]:.3f}, IDF={idf:.3f}, Source={source}, Freq={freq}")

    for phrase, freq in phrases:
        if freq < min_freq:
            continue
        tf = freq / total_words
        if phrase in IDF_APPROX:
            idf = IDF_APPROX[phrase]
            source = "JSON"
        else:
            idf = estimate_idf(phrase, phrase_freq, total_words, IDF_APPROX, KEYWORD_CATEGORIES, nlp)
            source = "Estimated"
        tfidf_scores[phrase] = tf * idf * tfidf_weight
        idf_sources[phrase] = {"idf": idf, "source": source, "frequency": freq}
        logger.debug(f"PDF term {phrase}: TF-IDF={tfidf_scores[phrase]:.3f}, IDF={idf:.3f}, Source={source}, Freq={freq}")

    # Apply attention mechanism: boost physics-related terms
    for term in tfidf_scores:
        for category, keywords in KEYWORD_CATEGORIES.items():
            if term in keywords and category in PHYSICS_CATEGORIES:
                tfidf_scores[term] *= 1.5  # Boost physics terms
                logger.debug(f"Boosted TF-IDF for {term}: {tfidf_scores[term]:.3f}")

    # Rank by TF-IDF if weight > 0, else by frequency
    if tfidf_weight > 0:
        ranked_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_limit]
    else:
        ranked_terms = [(w, f) for w, f in word_freq.most_common(top_limit) if f >= min_freq]
        ranked_terms += phrases[:top_limit - len(ranked_terms)]
    
    # Categorize keywords and phrases
    categorized_keywords = {cat: [] for cat in KEYWORD_CATEGORIES}
    term_to_category = {}
    for term, score in ranked_terms:
        if term in exclude_set:
            continue
        assigned = False
        for category, keywords in KEYWORD_CATEGORIES.items():
            if term in keywords:
                categorized_keywords[category].append((term, score))
                term_to_category[term] = category
                assigned = True
                break
            elif " " in term:
                if any(k == term or term.startswith(k + " ") or term.endswith(" " + k) for k in keywords):
                    categorized_keywords[category].append((term, score))
                    term_to_category[term] = category
                    assigned = True
                    break
        if not assigned:
            categorized_keywords["Other"].append((term, score))
            term_to_category[term] = "Other"
    
    logger.debug("Categorized keywords: %s", {k: [t[0] for t in v] for k, v in categorized_keywords.items()})
    return categorized_keywords, word_freq, phrases, tfidf_scores, term_to_category, idf_sources

def extract_text_from_pdf(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        pdf_reader = PyPDF2.PdfReader(tmp_file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        os.unlink(tmp_file_path)
        return text if text.strip() else "No text extracted from the PDF."
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return f"Error extracting text: {str(e)}"

def extract_text_between_phrases(text, start_phrase, end_phrase):
    try:
        start_idx = text.find(start_phrase)
        end_idx = text.find(end_phrase, start_idx + len(start_phrase))
        if start_idx == -1 or end_idx == -1:
            return "Specified phrases not found in the text."
        return text[start_idx:end_idx + len(end_phrase)]
    except Exception as e:
        logger.error(f"Error extracting text between phrases: {str(e)}")
        return f"Error extracting text between phrases: {str(e)}"

def clean_phrase(phrase, stop_words):
    words = phrase.split()
    # Remove leading stopwords (e.g., "the", "a", "an")
    while words and words[0].lower() in stop_words:
        words = words[1:]
    # Remove trailing stopwords
    while words and words[-1].lower() in stop_words:
        words = words[:-1]
    return " ".join(words).strip()

def generate_word_cloud(text, selected_keywords, tfidf_scores, selection_criteria, colormap, title_font_size, caption_font_size):
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure', 'the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
        stop_words.update([w.strip().lower() for w in custom_stopwords_input.split(",") if w.strip()])
        
        # Preserve phrases by replacing spaces with underscores
        processed_text = text.lower()
        keyword_map = {}
        for keyword in selected_keywords:
            internal_key = keyword.replace(" ", "_")
            processed_text = processed_text.replace(keyword, internal_key)
            keyword_map[internal_key] = keyword
        
        words = word_tokenize(processed_text)
        filtered_words = [keyword_map.get(word, word) for word in words if keyword_map.get(word, word) in selected_keywords]
        
        if not filtered_words:
            return None, "No valid words or phrases found for word cloud after filtering."
        
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            min_font_size=8,
            max_font_size=150,
            font_path=None,
            colormap=colormap
        ).generate_from_frequencies({word: tfidf_scores.get(word, 1.0) for word in filtered_words})
        
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud of Selected Keywords and Phrases", fontsize=title_font_size, pad=10)
        caption = f"Word Cloud generated with: {selection_criteria}"
        plt.figtext(
            0.5, 0.05, caption, ha="center", fontsize=caption_font_size, wrap=True,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"

def generate_bibliometric_network(
    text, selected_keywords, tfidf_scores, label_font_size, selection_criteria,
    node_colormap, edge_colormap, network_style, line_thickness, node_alpha, edge_alpha,
    title_font_size, caption_font_size, node_size_scale, node_shape, node_linewidth,
    node_edgecolor, edge_style, label_font_color, label_font_family, label_bbox_facecolor,
    label_bbox_alpha, layout_algorithm
):
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure', 'the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
        stop_words.update([w.strip().lower() for w in custom_stopwords_input.split(",") if w.strip()])
        
        # Preserve phrases by replacing spaces with underscores
        processed_text = text.lower()
        keyword_map = {}
        for keyword in selected_keywords:
            internal_key = keyword.replace(" ", "_")
            processed_text = processed_text.replace(keyword, internal_key)
            keyword_map[internal_key] = keyword
        
        words = word_tokenize(processed_text)
        filtered_words = [keyword_map.get(word, word) for word in words if keyword_map.get(word, word) in selected_keywords]
        
        word_freq = Counter(filtered_words)
        if not word_freq:
            return None, "No valid words or phrases found for bibliometric network."
        
        top_words = [word for word, freq in word_freq.most_common(20)]
        
        sentences = sent_tokenize(text.lower())
        co_occurrences = Counter()
        for sentence in sentences:
            processed_sentence = sentence
            for keyword in selected_keywords:
                processed_sentence = processed_sentence.replace(keyword, keyword.replace(" ", "_"))
            words_in_sentence = [keyword_map.get(word, word) for word in word_tokenize(processed_sentence) if keyword_map.get(word, word) in top_words]
            for pair in combinations(set(words_in_sentence), 2):
                co_occurrences[tuple(sorted(pair))] += 1
        
        G = nx.Graph()
        for word, freq in word_freq.most_common(20):
            G.add_node(word, size=freq)
        
        for (word1, word2), weight in co_occurrences.items():
            if word1 in top_words and word2 in top_words:
                G.add_edge(word1, word2, weight=weight)
        
        communities = greedy_modularity_communities(G)
        node_colors = {}
        try:
            cmap = plt.cm.get_cmap(node_colormap)
            palette = cmap(np.linspace(0, 1, max(1, len(communities))))
        except ValueError:
            logger.warning(f"Invalid node colormap {node_colormap}, falling back to viridis")
            cmap = plt.cm.get_cmap("viridis")
            palette = cmap(np.linspace(0, 1, max(1, len(communities))))
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = palette[i]
        
        edge_weights = [G.edges[edge]['weight'] for edge in G.edges]
        max_weight = max(edge_weights, default=1)
        edge_widths = [line_thickness * np.log1p(weight) + 0.5 for weight in edge_weights]
        
        try:
            edge_cmap = plt.cm.get_cmap(edge_colormap)
            edge_colors = [edge_cmap(weight / max_weight) for weight in edge_weights]
        except ValueError:
            logger.warning(f"Invalid edge colormap {edge_colormap}, falling back to Blues")
            edge_cmap = plt.cm.get_cmap("Blues")
            edge_colors = [edge_cmap(weight / max_weight) for weight in edge_weights]
        
        # Select layout algorithm
        try:
            if layout_algorithm == 'spring':
                pos = nx.spring_layout(G, k=0.6, seed=42)
            elif layout_algorithm == 'circular':
                pos = nx.circular_layout(G)
            elif layout_algorithm == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            elif layout_algorithm == 'shell':
                pos = nx.shell_layout(G)
            elif layout_algorithm == 'spectral':
                pos = nx.spectral_layout(G)
            elif layout_algorithm == 'random':
                pos = nx.random_layout(G, seed=42)
            elif layout_algorithm == 'spiral':
                pos = nx.spiral_layout(G)
            elif layout_algorithm == 'planar':
                try:
                    pos = nx.planar_layout(G)
                except nx.NetworkXException:
                    logger.warning("Graph is not planar, falling back to spring layout")
                    pos = nx.spring_layout(G, k=0.6, seed=42)
        except Exception as e:
            logger.error(f"Error in layout {layout_algorithm}: {str(e)}, falling back to spring")
            pos = nx.spring_layout(G, k=0.6, seed=42)
        
        try:
            plt.style.use(network_style)
        except ValueError:
            logger.warning(f"Invalid network style {network_style}, falling back to seaborn-v0_8-white")
            plt.style.use("seaborn-v0_8-white")
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        
        node_sizes = [G.nodes[node]['size'] * node_size_scale for node in G.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=[node_colors[node] for node in G.nodes],
            node_shape=node_shape,
            edgecolors=node_edgecolor,
            linewidths=node_linewidth,
            alpha=node_alpha,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            style=edge_style,
            alpha=edge_alpha,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=label_font_size,
            font_color=label_font_color,
            font_family=label_font_family,
            font_weight='bold',
            bbox=dict(
                facecolor=label_bbox_facecolor,
                alpha=label_bbox_alpha,
                edgecolor='none',
                boxstyle='round,pad=0.2'
            ),
            ax=ax
        )
        
        ax.set_title("Keyword Co-occurrence Network", fontsize=title_font_size, pad=15, fontweight='bold')
        caption = f"Keyword co-occurrence network generated with: {selection_criteria}"
        plt.figtext(
            0.5, 0.06, caption, ha="center", fontsize=caption_font_size, wrap=True,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        ax.set_facecolor('#f5f5f5')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating bibliometric network: {str(e)}")
        return None, f"Error generating bibliometric network: {str(e)}"

def generate_radar_chart(selected_keywords, values, title, selection_criteria, colormap, max_keywords, label_font_size, line_thickness, fill_alpha, title_font_size, caption_font_size):
    """
    Generate a radar chart for keyword/phrase values.
    Args:
        selected_keywords (list): List of selected keywords/phrases.
        values (Counter or dict): Frequency or TF-IDF scores.
        title (str): Chart title.
        selection_criteria (str): Criteria for caption.
        colormap (str): Matplotlib colormap name.
        max_keywords (int): Maximum keywords to display.
        label_font_size (int): Font size for axis labels.
        line_thickness (float): Thickness of radar lines.
        fill_alpha (float): Transparency of radar fill.
        title_font_size (int): Font size for title.
        caption_font_size (int): Font size for caption.
    Returns:
        tuple: (matplotlib figure, error message or None)
    """
    try:
        if len(selected_keywords) < 3:
            return None, "At least 3 keywords/phrases are required for a radar chart."
        
        # Select top keywords (up to max_keywords)
        keyword_values = [(k, values.get(k, 0)) for k in selected_keywords]
        keyword_values = sorted(keyword_values, key=lambda x: x[1], reverse=True)[:max_keywords]
        if not keyword_values:
            return None, "No valid keywords/phrases with values for radar chart."
        
        labels, vals = zip(*keyword_values)
        num_vars = len(labels)
        
        # Normalize values to [0, 1]
        max_val = max(vals, default=1)
        vals = [v / max_val for v in vals] if max_val > 0 else vals
        
        # Compute angles for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        vals = vals + [vals[0]]  # Close the plot
        angles += angles[:1]
        
        # Create polar plot
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw=dict(polar=True))
        
        # Apply colormap
        try:
            cmap = plt.cm.get_cmap(colormap)
            line_color = cmap(0.8)  # Vibrant color for line
            fill_color = cmap(0.6)  # Lighter color for fill
        except ValueError:
            logger.warning(f"Invalid radar colormap {colormap}, falling back to viridis")
            cmap = plt.cm.get_cmap("viridis")
            line_color = cmap(0.8)
            fill_color = cmap(0.6)
        
        # Plot data
        ax.plot(angles, vals, color=line_color, linewidth=line_thickness, linestyle='solid')
        ax.fill(angles, vals, color=fill_color, alpha=fill_alpha)
        
        # Customize axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=label_font_size, wrap=True)
        ax.set_rlabel_position(30)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Set title and caption with increased padding
        ax.set_title(title, fontsize=title_font_size, pad=25, fontweight='bold')
        caption = f"{title} generated with: {selection_criteria}"
        plt.figtext(
            0.5, 0.01, caption, ha="center", fontsize=caption_font_size, wrap=True,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout(rect=[0, 0.25, 1, 1])
        ax.set_facecolor('#f5f5f5')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating radar chart: {str(e)}")
        return None, f"Error generating radar chart: {str(e)}"

def save_figure(fig, filename):
    try:
        fig.savefig(filename + ".png", dpi=300, bbox_inches='tight', format='png')
        fig.savefig(filename + ".svg", bbox_inches='tight', format='svg')
        return True
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        return False

# Clear all selections button
if 'clear_selections' not in st.session_state:
    st.session_state.clear_selections = False

def clear_selections():
    st.session_state.clear_selections = True
    for key in list(st.session_state.keys()):
        if key.startswith("multiselect_"):
            del st.session_state[key]

# Set page configuration
st.set_page_config(page_title="PDF Text Extractor & Visualization for Additive Manufacturing", layout="wide")

# Title and description
st.title("PDF Text Extractor and Visualization for Additive Manufacturing")
st.markdown("""
Upload a PDF file to extract text between specified phrases, configure keyword and phrase selection criteria, and generate
publication-quality word clouds, bibliometric networks, and radar charts with customizable colormaps, styles, and layouts.
Adjust visual aspects like node sizes, edge styles, label properties, and transparency using sliders and dropdowns.
The app extracts text using PyPDF2, processes keywords and phrases with NLTK and spaCy, and visualizes using WordCloud,
NetworkX, Matplotlib, and Seaborn.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Input fields for desired phrases and stopwords
start_phrase = st.text_input("Enter the desired initial phrase", "Introduction")
end_phrase = st.text_input("Enter the desired final phrase", "Conclusion")
custom_stopwords_input = st.text_input("Custom stopwords (comma-separated)", "et al,figure,table", help="Words to exclude from keywords/phrases")
exclude_keywords_input = st.text_input("Exclude keywords/phrases (comma-separated)", "preprint,submitted,manuscript", help="Keywords/phrases to remove from categories")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        
        if "Error" in text:
            st.error(text)
        else:
            selected_text = extract_text_between_phrases(text, start_phrase, end_phrase)
            
            if "Error" in selected_text or "not found" in selected_text:
                st.error(selected_text)
            else:
                st.subheader("Extracted Text Between Phrases")
                st.text_area("Selected Text", selected_text, height=200)
                
                st.subheader("Configure Keyword Selection Criteria")
                min_freq = st.slider("Minimum frequency", min_value=1, max_value=10, value=1, help="Minimum occurrences of a word/phrase")
                min_length = st.slider("Minimum length", min_value=3, max_value=30, value=10, help="Minimum characters in a word/phrase")
                use_stopwords = st.checkbox("Use stopword filtering", value=True, help="Remove common English words (e.g., 'the', 'is')")
                top_limit = st.slider("Top limit (max keywords)", min_value=10, max_value=100, value=50, step=10, help="Maximum number of candidate keywords")
                tfidf_weight = st.slider("TF-IDF weighting (statistical relevance)", min_value=0.0, max_value=1.0, value=1.0, step=0.1, help="Higher values prioritize rare, significant terms")
                use_nouns_only = st.checkbox("Filter for nouns only (linguistic filtering)", value=False, help="Include only nouns for more specific terms")
                include_phrases = st.checkbox("Include multi-word phrases", value=True, help="Extract noun phrases like 'selective laser melting'", disabled=True)
                
                criteria_parts = [
                    f"frequency ≥ {min_freq}",
                    f"length ≥ {min_length}",
                    "stopwords " + ("enabled" if use_stopwords else "disabled"),
                    f"custom stopwords: {custom_stopwords_input}" if custom_stopwords_input.strip() else "no custom stopwords",
                    f"excluded keywords: {exclude_keywords_input}" if exclude_keywords_input.strip() else "no excluded keywords",
                    f"top {top_limit} keywords",
                    f"TF-IDF weight: {tfidf_weight}",
                    "nouns only" if use_nouns_only else "all parts of speech",
                    "multi-word phrases included"
                ]
                
                st.subheader("Select Keywords and Phrases by Category")
                if st.button("Clear All Selections"):
                    clear_selections()
                
                try:
                    categorized_keywords, word_freq, phrases, tfidf_scores, term_to_category, idf_sources = get_candidate_keywords(
                        selected_text, min_freq, min_length, use_stopwords, custom_stopwords_input, exclude_keywords_input, top_limit, tfidf_weight, use_nouns_only, include_phrases
                    )
                except Exception as e:
                    st.error(f"Error processing keywords: {str(e)}")
                    logger.error(f"Error in get_candidate_keywords: {str(e)}")
                    st.stop()
                
                selected_keywords = []
                for category in KEYWORD_CATEGORIES:
                    keywords = [term for term, _ in categorized_keywords.get(category, [])]
                    with st.expander(f"{category} ({len(keywords)} keywords/phrases)"):
                        if keywords:
                            selected = st.multiselect(
                                f"Select keywords from {category}",
                                options=keywords,
                                default=[] if st.session_state.clear_selections else keywords[:min(5, len(keywords))],
                                key=f"multiselect_{category}_{uuid.uuid4()}"
                            )
                            selected_keywords.extend(selected)
                        else:
                            st.write("No keywords or phrases found for this category. Try lowering min_freq or min_length.")
                
                # Reset clear_selections flag
                st.session_state.clear_selections = False
                
                # Debug information
                with st.expander("Debug Information"):
                    if 'word_freq' in locals() and word_freq:
                        st.write("Single Words (Top 20):", word_freq.most_common(20))
                    else:
                        st.write("Single Words: Not available yet.")
                    if 'phrases' in locals() and phrases:
                        st.write("Extracted Phrases (Top 20):", phrases[:20])
                    else:
                        st.write("Extracted Phrases: Not available yet.")
                    if 'categorized_keywords' in locals() and categorized_keywords:
                        st.write("Categorized Keywords:", {k: [t[0] for t in v] for k, v in categorized_keywords.items()})
                    else:
                        st.write("Categorized Keywords: Not available yet.")
                
                # IDF Source Details with highlighting and filter
                with st.expander("IDF Source Details"):
                    if 'idf_sources' in locals() and idf_sources:
                        # Create DataFrame
                        idf_data = [
                            {
                                "Term": term,
                                "Frequency": idf_sources[term]["frequency"],
                                "TF-IDF Score": round(tfidf_scores.get(term, 0), 3),
                                "IDF Value": round(idf_sources[term]["idf"], 3),
                                "Source": idf_sources[term]["source"]
                            }
                            for term in tfidf_scores
                        ]
                        idf_df = pd.DataFrame(idf_data).sort_values(by=["Source", "TF-IDF Score"], ascending=[True, False])
                        
                        # Style DataFrame: Bold JSON sources
                        def highlight_json(row):
                            if row["Source"] == "JSON":
                                return ["font-weight: bold"] * len(row)
                            return [""] * len(row)
                        
                        # Filter by Source
                        source_filter = st.selectbox(
                            "Filter by IDF Source",
                            ["All", "JSON", "Estimated"],
                            help="Show all terms, only JSON (from idf_approx.json), or only Estimated (from estimate_idf)"
                        )
                        if source_filter != "All":
                            idf_df = idf_df[idf_df["Source"] == source_filter]
                        
                        st.write("IDF Sources for PDF-Extracted Keywords/Phrases (JSON = idf_approx.json, Estimated = computed via estimate_idf, bold = JSON):")
                        styled_df = idf_df.style.apply(highlight_json, axis=1).format(
                            {"TF-IDF Score": "{:.3f}", "IDF Value": "{:.3f}"}
                        )
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Download IDF sources
                        st.download_button(
                            label="Download IDF Sources (JSON)",
                            data=json.dumps(idf_data, indent=4),
                            file_name="idf_sources.json",
                            mime="application/json"
                        )
                        
                        # Show estimated IDF cache
                        if 'custom_idf' in st.session_state:
                            st.write("Estimated IDF Cache:", st.session_state.custom_idf)
                    else:
                        st.write("IDF Sources: Not available yet.")
                
                if not selected_keywords:
                    st.error("Please select at least one keyword or phrase.")
                    st.stop()
                
                st.subheader("Visualization Settings")
                st.markdown("### General Visualization Settings")
                label_font_size = st.slider(
                    "Label font size (network nodes, radar axes)",
                    min_value=8, max_value=20, value=10, step=1,
                    help="Adjust font size for axis labels in network and radar charts"
                )
                line_thickness = st.slider(
                    "Line thickness (network edges, radar lines)",
                    min_value=0.5, max_value=5.0, value=2.0, step=0.5,
                    help="Adjust thickness of edges in network and lines in radar charts"
                )
                title_font_size = st.slider(
                    "Title font size (all visualizations)",
                    min_value=10, max_value=20, value=14, step=1,
                    help="Adjust font size for titles in all visualizations"
                )
                caption_font_size = st.slider(
                    "Caption font size (all visualizations)",
                    min_value=8, max_value=14, value=10, step=1,
                    help="Adjust font size for captions in all visualizations"
                )
                transparency = st.slider(
                    "Transparency (network nodes/edges, radar fills)",
                    min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                    help="Adjust transparency for network nodes/edges and radar chart fills"
                )
                criteria_parts.extend([
                    f"label font size: {label_font_size}",
                    f"line thickness: {line_thickness}",
                    f"title font size: {title_font_size}",
                    f"caption font size: {caption_font_size}",
                    f"transparency: {transparency}"
                ])
                
                st.markdown("### Word Cloud Settings")
                wordcloud_colormap = st.selectbox(
                    "Select colormap for word cloud",
                    options=COLORMAPS,
                    index=0,
                    help="Choose a colormap for the word cloud text colors"
                )
                criteria_parts.append(f"word cloud colormap: {wordcloud_colormap}")
                
                st.markdown("### Network Settings")
                network_style = st.selectbox(
                    "Select style for network",
                    options=NETWORK_STYLES,
                    index=0,
                    help="Choose a visual style for the network (e.g., ggplot, classic)"
                )
                node_colormap = st.selectbox(
                    "Select colormap for network nodes",
                    options=COLORMAPS,
                    index=0,
                    help="Choose a colormap for the node colors (community-based)"
                )
                edge_colormap = st.selectbox(
                    "Select colormap for network edges",
                    options=COLORMAPS,
                    index=0,
                    help="Choose a colormap for the edge colors (weight-based)"
                )
                layout_algorithm = st.selectbox(
                    "Select layout algorithm for network",
                    options=LAYOUT_ALGORITHMS,
                    index=0,
                    help="Choose a layout algorithm for node positioning (e.g., spring, circular)"
                )
                st.markdown("#### Node Settings")
                node_size_scale = st.slider(
                    "Node size scale",
                    min_value=10, max_value=100, value=50, step=5,
                    help="Adjust the scaling factor for node sizes based on frequency"
                )
                node_shape = st.selectbox(
                    "Select node shape",
                    options=NODE_SHAPES,
                    index=0,
                    help="Choose the shape for nodes (e.g., 'o' for circle, 's' for square)"
                )
                node_linewidth = st.slider(
                    "Node border thickness",
                    min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                    help="Adjust the thickness of node borders"
                )
                node_edgecolor = st.selectbox(
                    "Select node border color",
                    options=COLORS,
                    index=0,
                    help="Choose the color for node borders"
                )
                st.markdown("#### Edge Settings")
                edge_style = st.selectbox(
                    "Select edge style",
                    options=EDGE_STYLES,
                    index=0,
                    help="Choose the style for edges (e.g., solid, dashed)"
                )
                st.markdown("#### Label Settings")
                label_font_color = st.selectbox(
                    "Select label font color",
                    options=COLORS,
                    index=0,
                    help="Choose the color for node labels"
                )
                label_font_family = st.selectbox(
                    "Select label font family",
                    options=FONT_FAMILIES,
                    index=0,
                    help="Choose the font family for node labels"
                )
                label_bbox_facecolor = st.selectbox(
                    "Select label background color",
                    options=BBOX_COLORS,
                    index=0,
                    help="Choose the background color for label boxes"
                )
                label_bbox_alpha = st.slider(
                    "Label background transparency",
                    min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                    help="Adjust transparency of label background boxes"
                )
                criteria_parts.extend([
                    f"network style: {network_style}",
                    f"node colormap: {node_colormap}",
                    f"edge colormap: {edge_colormap}",
                    f"layout: {layout_algorithm}",
                    f"node size scale: {node_size_scale}",
                    f"node shape: {node_shape}",
                    f"node border thickness: {node_linewidth}",
                    f"node border color: {node_edgecolor}",
                    f"edge style: {edge_style}",
                    f"label font color: {label_font_color}",
                    f"label font family: {label_font_family}",
                    f"label background color: {label_bbox_facecolor}",
                    f"label background transparency: {label_bbox_alpha}"
                ])
                
                st.markdown("### Radar Chart Settings")
                radar_max_keywords = st.slider(
                    "Number of keywords for radar charts",
                    min_value=3, max_value=10, value=5, step=1,
                    help="Select how many keywords/phrases to display (3–10)"
                )
                freq_radar_colormap = st.selectbox(
                    "Select colormap for frequency radar chart",
                    options=COLORMAPS,
                    index=0,
                    help="Choose a colormap for the frequency radar chart"
                )
                tfidf_radar_colormap = st.selectbox(
                    "Select colormap for TF-IDF radar chart",
                    options=COLORMAPS,
                    index=0,
                    help="Choose a colormap for the TF-IDF radar chart"
                )
                criteria_parts.extend([
                    f"frequency radar colormap: {freq_radar_colormap}",
                    f"tfidf radar colormap: {tfidf_radar_colormap}"
                ])
                
                selection_criteria = ", ".join(criteria_parts)
                
                st.subheader("Word Cloud")
                wordcloud_fig, wordcloud_error = generate_word_cloud(
                    selected_text, selected_keywords, tfidf_scores, selection_criteria,
                    wordcloud_colormap, title_font_size, caption_font_size
                )
                if wordcloud_error:
                    st.error(wordcloud_error)
                elif wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                    if save_figure(wordcloud_fig, "wordcloud"):
                        st.download_button(
                            label="Download Word Cloud (PNG)",
                            data=open("wordcloud.png", "rb").read(),
                            file_name="wordcloud.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Word Cloud (SVG)",
                            data=open("wordcloud.svg", "rb").read(),
                            file_name="wordcloud.svg",
                            mime="image/svg+xml"
                        )
                
                st.subheader("Bibliometric Network")
                network_fig, network_error = generate_bibliometric_network(
                    selected_text, selected_keywords, tfidf_scores, label_font_size, selection_criteria,
                    node_colormap, edge_colormap, network_style, line_thickness, transparency, transparency,
                    title_font_size, caption_font_size, node_size_scale, node_shape, node_linewidth,
                    node_edgecolor, edge_style, label_font_color, label_font_family, label_bbox_facecolor,
                    label_bbox_alpha, layout_algorithm
                )
                if network_error:
                    st.error(network_error)
                elif network_fig:
                    st.pyplot(network_fig)
                    if save_figure(network_fig, "network"):
                        st.download_button(
                            label="Download Network (PNG)",
                            data=open("network.png", "rb").read(),
                            file_name="network.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Network (SVG)",
                            data=open("network.svg", "rb").read(),
                            file_name="network.svg",
                            mime="image/svg+xml"
                        )
                
                st.subheader("Frequency Radar Chart")
                freq_radar_fig, freq_radar_error = generate_radar_chart(
                    selected_keywords, word_freq, "Keyword/Phrase Frequency Comparison",
                    selection_criteria, freq_radar_colormap, radar_max_keywords,
                    label_font_size, line_thickness, transparency, title_font_size, caption_font_size
                )
                if freq_radar_error:
                    st.error(freq_radar_error)
                elif freq_radar_fig:
                    st.pyplot(freq_radar_fig)
                    if save_figure(freq_radar_fig, "freq_radar"):
                        st.download_button(
                            label="Download Frequency Radar (PNG)",
                            data=open("freq_radar.png", "rb").read(),
                            file_name="freq_radar.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Frequency Radar (SVG)",
                            data=open("freq_radar.svg", "rb").read(),
                            file_name="freq_radar.svg",
                            mime="image/svg+xml"
                        )
                
                st.subheader("TF-IDF Radar Chart")
                tfidf_radar_fig, tfidf_radar_error = generate_radar_chart(
                    selected_keywords, tfidf_scores, "Keyword/Phrase TF-IDF Comparison",
                    selection_criteria, tfidf_radar_colormap, radar_max_keywords,
                    label_font_size, line_thickness, transparency, title_font_size, caption_font_size
                )
                if tfidf_radar_error:
                    st.error(tfidf_radar_error)
                elif tfidf_radar_fig:
                    st.pyplot(tfidf_radar_fig)
                    if save_figure(tfidf_radar_fig, "tfidf_radar"):
                        st.download_button(
                            label="Download TF-IDF Radar (PNG)",
                            data=open("tfidf_radar.png", "rb").read(),
                            file_name="tfidf_radar.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download TF-IDF Radar (SVG)",
                            data=open("tfidf_radar.svg", "rb").read(),
                            file_name="tfidf_radar.svg",
                            mime="image/svg+xml"
                        )

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, PyPDF2, WordCloud, NetworkX, NLTK, spaCy, Matplotlib, and Seaborn for additive manufacturing research.")
