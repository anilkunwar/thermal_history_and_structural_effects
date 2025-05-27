import streamlit as st
import arxiv
import math
import json
import logging
import pandas as pd
import time
from collections import Counter
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Single Term TF-IDF Calculator",
    layout="wide"
)

# Title and description
st.title("TF-IDF Calculator for Additive Manufacturing")
st.markdown("""
This Streamlit app computes the Term Frequency (TF), Inverse Document Frequency (IDF), and TF-IDF
for a single keyword or phrase in the context of additive manufacturing research using the arXiv database.
Enter a term or phrase, and the app will query arXiv to estimate TF (based on term occurrences in abstracts),
IDF (based on document frequency), and TF-IDF in materials science and applied physics categories.
The actual number of documents scanned (N) is dynamically estimated from the arXiv database.
The result is displayed and saved as a JSON file.
""")

# Function to estimate corpus size (N)
def estimate_corpus_size():
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query="cat:cond-mat.mtrl-sci OR cat:physics.app-ph",
            max_results=0  # Use max_results=0 to get total count without fetching results
        )
        total_results = sum(1 for _ in search.results())  # This will be 0, so we need metadata
        # Perform a lightweight query to get total results
        search = arxiv.Search(
            query="cat:cond-mat.mtrl-sci OR cat:physics.app-ph",
            max_results=1
        )
        results = list(search.results())
        if results:
            # Estimate total by querying a small sample and using metadata (not directly available)
            # Since arXiv doesn't provide exact total, we use a large sample for estimation
            search = arxiv.Search(
                query="cat:cond-mat.mtrl-sci OR cat:physics.app-ph",
                max_results=1000
            )
            N = sum(1 for _ in search.results())
            # Scale up based on typical arXiv category sizes (heuristic)
            N = N * 100  # Rough scaling factor based on arXiv's large corpus
            return max(N, 1)  # Avoid division by zero
        return 100000  # Fallback if query fails
    except Exception as e:
        logger.error(f"Error estimating corpus size: {str(e)}")
        return 100000  # Fallback to original estimate

# Estimate corpus size (cached to avoid repeated queries)
if 'corpus_size' not in st.session_state:
    with st.spinner("Estimating corpus size..."):
        st.session_state.corpus_size = estimate_corpus_size()
        time.sleep(1)  # Respect rate limits
N = st.session_state.corpus_size
st.write(f"**Estimated corpus size (N)**: {N} documents (materials science and applied physics)")

# Input field for keyword/phrase
term = st.text_input("Enter a keyword or phrase", value="selective laser melting", help="E.g., 'microscopy', 'bimodal microstructure'")

# Function to compute TF from abstracts
def compute_tf(term, abstracts):
    if not abstracts:
        return 0.0
    term = term.lower()
    total_words = 0
    term_count = 0
    for abstract in abstracts:
        words = re.findall(r'\w+', abstract.lower())
        total_words += len(words)
        term_count += len([w for w in words if w == term or term in ' '.join(words)])
    return term_count / total_words if total_words > 0 else 0.0

# Button to compute TF-IDF
if st.button("Compute TF-IDF"):
    if not term.strip():
        st.error("Please enter a valid keyword or phrase.")
    else:
        with st.spinner(f"Querying arXiv for '{term}'..."):
            try:
                # Format query (wrap phrases in quotes)
                query_term = f'"{term}"' if " " in term else term
                query = f"{query_term} cat:cond-mat.mtrl-sci OR cat:physics.app-ph"
                client = arxiv.Client()
                search = arxiv.Search(
                    query=query,
                    max_results=1000,  # Limit to avoid timeout
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                # Collect abstracts and count documents
                abstracts = []
                doc_count = 0
                for result in search.results():
                    abstracts.append(result.summary)
                    doc_count += 1
                doc_count = max(doc_count, 1)  # Avoid division by zero
                
                # Compute IDF
                idf = math.log(N / doc_count)
                
                # Compute TF
                tf = compute_tf(term, abstracts)
                
                # Compute TF-IDF
                tf_idf = tf * idf
                
                logger.info(f"Term: {term}, Documents: {doc_count}, TF: {tf:.3f}, IDF: {idf:.3f}, TF-IDF: {tf_idf:.3f}")

                # Prepare result
                result = [{
                    "Term": term,
                    "Document Count": doc_count,
                    "TF": round(tf, 3),
                    "IDF": round(idf, 3),
                    "TF-IDF": round(tf_idf, 3)
                }]
                df = pd.DataFrame(result)
                
                # Save to JSON
                output_path = "tf_idf_single_term.json"
                tf_idf_dict = {term: {"TF": tf, "IDF": idf, "TF-IDF": tf_idf, "Corpus Size (N)": N}}
                try:
                    with open(output_path, "w") as f:
                        json.dump(tf_idf_dict, f, indent=4)
                    logger.info(f"TF-IDF saved to {output_path}")
                    st.success(f"TF-IDF saved to `{output_path}`")
                except Exception as e:
                    logger.error(f"Error saving JSON: {str(e)}")
                    st.error(f"Error saving JSON: {str(e)}")

                # Display result
                st.subheader("TF-IDF Result")
                st.dataframe(df, use_container_width=True)

                # Download button
                with open(output_path, "r") as f:
                    st.download_button(
                        label="Download tf_idf_single_term.json",
                        data=f,
                        file_name="tf_idf_single_term.json",
                        mime="application/json"
                    )

            except Exception as e:
                logger.error(f"Error querying '{term}': {str(e)}")
                st.error(f"Error computing TF-IDF: {str(e)}")
                result = [{
                    "Term": term,
                    "Document Count": "Error",
                    "TF": "N/A",
                    "IDF": "N/A",
                    "TF-IDF": "N/A"
                }]
                st.subheader("TF-IDF Result")
                st.dataframe(pd.DataFrame(result), use_container_width=True)

            # Respect arXiv rate limits
            time.sleep(1)

# Instructions for integration
st.subheader("Integration Instructions")
st.markdown("""
1. **Download `tf_idf_single_term.json`** after computation.
2. Use the TF, IDF, TF-IDF, and Corpus Size (N) values in your application. Example:
   ```python
   import json
   with open("tf_idf_single_term.json", "r") as f:
       tf_idf_dict = json.load(f)
   term = "selective laser melting"
   metrics = tf_idf_dict.get(term, {"TF": 0.0, "IDF": math.log(100000 / 10000), "TF-IDF": 0.0, "Corpus Size (N)": 100000})
   print(f"Corpus Size (N): {metrics['Corpus Size (N)']}")
   print(f"TF for {term}: {metrics['TF']:.3f}")
   print(f"IDF for {term}: {metrics['IDF']:.3f}")
   print(f"TF-IDF for {term}: {metrics['TF-IDF']:.3f}")
   ```
3. For multiple terms, compute each TF-IDF separately or modify the app to batch process terms.
""")

# Footer
st.markdown("---")
st.markdown("Compute TF, IDF, and TF-IDF for a single term or phrase in additive manufacturing research using arXiv.")
