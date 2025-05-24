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

# Simplified IDF approximation for scientific texts
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
DEFAULT_IDF = log(1000 / 100)

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
        "non-uniform temperature", "laser-material interaction", "melt boundary", "processing parameters"
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

# Set page configuration
st.set_page_config(page_title="PDF Text Extractor & Visualization for Additive Manufacturing", layout="wide")

# Title and description
st.title("Lightweight Attention Mechanism aided Text Visualization for Additive Manufacturing")
st.markdown("""
Upload a PDF file to extract text between specified phrases, configure keyword and phrase selection criteria, and generate
publication-quality word clouds and colorful bibliometric networks with customizable colormaps. The app extracts text using PyPDF2,
processes keywords and phrases with NLTK and spaCy, and visualizes using WordCloud and NetworkX.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Input fields for desired phrases and stopwords
start_phrase = st.text_input("Enter the desired initial phrase", "Introduction")
end_phrase = st.text_input("Enter the desired final phrase", "Conclusion")
custom_stopwords_input = st.text_input("Custom stopwords (comma-separated)", "et al,figure,table", help="Words to exclude from keywords/phrases")
exclude_keywords_input = st.text_input("Exclude keywords/phrases (comma-separated)", "preprint,submitted,manuscript", help="Keywords/phrases to remove from categories")

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

    # Compute TF-IDF scores with attention mechanism
    total_words = len(word_tokenize(text))
    tfidf_scores = {}
    for word, freq in word_freq.items():
        if freq < min_freq:
            continue
        tf = freq / total_words
        idf = IDF_APPROX.get(word, DEFAULT_IDF)
        tfidf_scores[word] = tf * idf * tfidf_weight

    for phrase, freq in phrases:
        if freq < min_freq:
            continue
        tf = freq / total_words
        idf = IDF_APPROX.get(phrase, DEFAULT_IDF)
        tfidf_scores[phrase] = tf * idf * tfidf_weight

    # Apply attention mechanism: boost physics-related terms
    for term in tfidf_scores:
        for category, keywords in KEYWORD_CATEGORIES.items():
            if term in keywords and category in PHYSICS_CATEGORIES:
                tfidf_scores[term] *= 1.5  # Boost physics terms
                logger.debug(f"Boosted TF-IDF for {term}: {tfidf_scores[term]}")

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
    return categorized_keywords, word_freq, phrases, tfidf_scores, term_to_category

def generate_word_cloud(text, selected_keywords, tfidf_scores, selection_criteria, colormap):
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
        ax.set_title("Word Cloud of Selected Keywords and Phrases", fontsize=14, pad=10)
        caption = f"Word Cloud generated with: {selection_criteria}"
        plt.figtext(0.5, 0.01, caption, ha="center", fontsize=10, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"

def generate_bibliometric_network(text, selected_keywords, tfidf_scores, label_font_size, selection_criteria, node_colormap, edge_colormap):
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
        edge_widths = [2.5 * np.log1p(weight) + 1 for weight in edge_weights]
        try:
            edge_cmap = plt.cm.get_cmap(edge_colormap)
            edge_colors = [edge_cmap(weight / max_weight) for weight in edge_weights]
        except ValueError:
            logger.warning(f"Invalid edge colormap {edge_colormap}, falling back to Blues")
            edge_cmap = plt.cm.get_cmap("Blues")
            edge_colors = [edge_cmap(weight / max_weight) for weight in edge_weights]
        
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        pos = nx.spring_layout(G, k=0.6, seed=42)
        
        node_sizes = [G.nodes[node]['size'] * 50 for node in G.nodes]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=[node_colors[node] for node in G.nodes],
            edgecolors='black',
            linewidths=1.0,
            alpha=0.9,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            G, pos,
            font_size=label_font_size,
            font_weight='bold',
            font_color='white',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
            ax=ax
        )
        
        ax.set_title("Keyword Co-occurrence Network", fontsize=16, pad=15, fontweight='bold')
        caption = f"Keyword co-occurrence network generated with: {selection_criteria}"
        plt.figtext(
            0.5, 0.02,
            caption,
            ha="center",
            fontsize=10,
            wrap=True,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        ax.set_facecolor('#f5f5f5')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating bibliometric network: {str(e)}")
        return None, f"Error generating bibliometric network: {str(e)}"

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
                    categorized_keywords, word_freq, phrases, tfidf_scores, term_to_category = get_candidate_keywords(
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
                
                if not selected_keywords:
                    st.error("Please select at least one keyword or phrase.")
                    st.stop()
                
                st.subheader("Visualization Settings")
                st.markdown("### Word Cloud Settings")
                wordcloud_colormap = st.selectbox(
                    "Select colormap for word cloud",
                    options=COLORMAPS,
                    index=0,
                    help="Choose a colormap for the word cloud text colors"
                )
                criteria_parts.append(f"word cloud colormap: {wordcloud_colormap}")
                
                st.markdown("### Network Settings")
                label_font_size = st.slider("Select font size for network labels", min_value=8, max_value=20, value=10, step=1)
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
                criteria_parts.append(f"node colormap: {node_colormap}")
                criteria_parts.append(f"edge colormap: {edge_colormap}")
                
                selection_criteria = ", ".join(criteria_parts)
                
                st.subheader("Word Cloud")
                wordcloud_fig, wordcloud_error = generate_word_cloud(selected_text, selected_keywords, tfidf_scores, selection_criteria, wordcloud_colormap)
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
                network_fig, network_error = generate_bibliometric_network(selected_text, selected_keywords, tfidf_scores, label_font_size, selection_criteria, node_colormap, edge_colormap)
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

# Footer
st.markdown("---")
st.markdown("Philology in Additive Manufacturing.")
