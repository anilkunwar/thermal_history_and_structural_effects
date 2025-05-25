import streamlit as st
import arxiv
import math
import json
import logging
import pandas as pd
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Single Term IDF Calculator",
    layout="wide"
)

# Title and description
st.title("Inverse Document Frequency Calculator for Additive Manufacturing")
st.markdown("""
This Streamlit app computes the Inverse Document Frequency (IDF) for a single keyword or phrase
in the context of additive manufacturing research using the arXiv database. Enter a term or phrase,
and the app will query arXiv to estimate its IDF based on document frequency in materials science
and applied physics categories. The result is displayed and saved as a JSON file.
""")

# Corpus size
N = 100000  # Approx. materials science/engineering papers
st.write(f"**Estimated corpus size (N)**: {N} documents (materials science and applied physics)")

# Input field for keyword/phrase
term = st.text_input("Enter a keyword or phrase", value="selective laser melting", help="E.g., 'microscopy', 'bimodal microstructure'")

# Button to compute IDF
if st.button("Compute IDF"):
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
                # Count results
                doc_count = sum(1 for _ in search.results())
                doc_count = max(doc_count, 1)  # Avoid division by zero
                idf = math.log(N / doc_count)
                logger.info(f"Term: {term}, Documents: {doc_count}, IDF: {idf:.3f}")

                # Prepare result
                result = [{"Term": term, "Document Count": doc_count, "IDF": round(idf, 3)}]
                df = pd.DataFrame(result)
                
                # Save to JSON
                output_path = "idf_single_term.json"
                idf_dict = {term: idf}
                try:
                    with open(output_path, "w") as f:
                        json.dump(idf_dict, f, indent=4)
                    logger.info(f"IDF saved to {output_path}")
                    st.success(f"IDF saved to `{output_path}`")
                except Exception as e:
                    logger.error(f"Error saving JSON: {str(e)}")
                    st.error(f"Error saving JSON: {str(e)}")

                # Display result
                st.subheader("IDF Result")
                st.dataframe(df, use_container_width=True)

                # Download button
                with open(output_path, "r") as f:
                    st.download_button(
                        label="Download idf_single_term.json",
                        data=f,
                        file_name="idf_single_term.json",
                        mime="application/json"
                    )

            except Exception as e:
                logger.error(f"Error querying '{term}': {str(e)}")
                st.error(f"Error computing IDF: {str(e)}")
                result = [{"Term": term, "Document Count": "Error", "IDF": "N/A"}]
                st.subheader("IDF Result")
                st.dataframe(pd.DataFrame(result), use_container_width=True)

            # Respect arXiv rate limits
            time.sleep(1)

# Instructions for integration
st.subheader("Integration Instructions")
st.markdown("""
1. **Download `idf_single_term.json`** after computation.
2. Use the IDF value in your application. Example:
   ```python
   import json
   with open("idf_single_term.json", "r") as f:
       idf_dict = json.load(f)
   term = "selective laser melting"
   idf = idf_dict.get(term, math.log(100000 / 10000))  # Fallback IDF
   print(f"IDF for {term}: {idf:.3f}")
   ```
3. For multiple terms, compute each IDF separately or modify the app to batch process terms.
""")

# Footer
st.markdown("---")
st.markdown("Compute IDF for a single term or phrase in additive manufacturing research using arXiv.")