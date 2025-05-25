import streamlit as st
import arxiv
import math
import json
import time
import logging
from collections import Counter
import pandas as pd

# Set up logging (consistent with your main app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Data-driven study of keywords-keyphrases",
    layout="wide"
)

# Title and description
st.title("Inverse Document Frequency Calculation for Selected Keywords-Keyphrases of Additive Manufacturing ")
st.markdown("""
This Streamlit app computes Inverse Document Frequency (IDF) values for terms and phrases
from the arXiv database, tailored for additive manufacturing research. It queries arXiv for
document counts, calculates IDFs, and generates a downloadable `idf_approx.json` file to
replace the hardcoded `IDF_APPROX` table in your main app. The app focuses on terms from
`IDF_APPROX` and `KEYWORD_CATEGORIES`, ensuring relevance to materials science and engineering.
""")

# Define terms from your app's IDF_APPROX
IDF_APPROX_TERMS = [
    "study", "analysis", "results", "method", "experiment",
    "spectroscopy", "nanoparticle", "diffraction", "microscopy", "quantum",
    "selective laser melting", "bimodal microstructure", "stacking faults",
    "al-si-mg", "strength-ductility", "finite element analysis",
    "molecular dynamics", "scanning electron microscopy",
    "transmission electron microscopy", "slm-fabricated alsimg1.4zr",
    "grain refinement", "dislocation dynamics", "heterogeneous nucleation",
    "melt pool dynamics", "thermal gradient"
]

# Define terms from KEYWORD_CATEGORIES (flattened and deduplicated)
KEYWORD_CATEGORIES_TERMS = [
    # Materials
    "alloy", "polymer", "nanoparticle", "crystal", "metal", "ceramic",
    "composite", "semiconductor", "graphene", "nanotube", "oxide",
    "thin film", "superconductor", "biomaterial", "aluminum", "magnesium",
    "zirconium", "silicon", "intermetallic", "al3zr", "aluminum alloy",
    "al-si-mg", "alsimg1.4zr", "lightweight alloy", "high-mg-content",
    "zr-modified alloy", "slm-fabricated alsimg1.4zr",
    # Methods
    "microscopy", "lithography", "deposition", "etching", "annealing",
    "characterization", "synthesis", "fabrication", "imaging", "scanning",
    "tomography", "spectroscopy", "diffraction", "ablation", "crystallization",
    "polymerization", "evaporation", "sputtering", "measurement", "aging",
    "electropolishing", "ion-beam thinning", "tensile testing",
    "selective laser melting", "vacuum induction", "gas atomization",
    "finite element analysis", "molecular dynamics",
    "electron backscatter diffraction", "energy-dispersive x-ray",
    "x-ray diffraction", "scanning electron microscopy",
    "transmission electron microscopy", "direct aging", "bidirectional scanning",
    "laser processing", "thermomechanical treatment",
    "severe plastic deformation",
    # Physical Phenomena
    "diffusion", "scattering", "conductivity", "magnetism", "superconductivity",
    "fluorescence", "polarization", "refraction", "absorption", "emission",
    "quantum", "thermal", "solidification", "deformation", "nucleation",
    "dislocation", "twinning", "stacking fault", "thermal gradient",
    "stacking faults", "dislocation dynamics", "grain refinement",
    "heterogeneous nucleation", "bimodal microstructure", "melt pool dynamics",
    "rapid cooling", "thermal history",
    # Properties
    "hardness", "conductivity", "resistivity", "magnetization", "density",
    "strength", "elasticity", "viscosity", "porosity", "permeability",
    "ductility", "toughness", "plasticity", "elongation", "yield strength",
    "ultimate tensile strength", "work hardening", "strength-ductility",
    "grain size", "thermal expansion", "stacking fault energy",
    "mechanical properties", "bimodal grains", "hall-petch",
    "work hardening rate", "strain hardening",
    # Other
    "sustainability", "industry 4.0", "optimization", "simulation", "modeling",
    "energy efficiency", "additive manufacturing", "sustainable manufacturing",
    "process-structure", "hierarchical microstructure",
    "non-uniform temperature", "laser-material interaction", "melt boundary"
]

# Combine and deduplicate terms
terms = list(set(IDF_APPROX_TERMS + KEYWORD_CATEGORIES_TERMS))
logger.info(f"Total unique terms to query: {len(terms)}")
st.write(f"**Total unique terms to query**: {len(terms)}")

# Corpus size (estimated for additive manufacturing in arXiv)
N = 100000  # Approx. materials science/engineering papers
st.write(f"**Estimated corpus size (N)**: {N} documents (materials science and applied physics)")

# Initialize IDF dictionary
IDF_APPROX = {}

# Button to start computation
if st.button("Compute IDFs from arXiv"):
    with st.spinner("Querying arXiv and computing IDFs..."):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        # Query arXiv
        client = arxiv.Client()
        for i, term in enumerate(terms):
            try:
                # Format query (wrap phrases in quotes)
                query_term = f'"{term}"' if " " in term else term
                # Search in materials science and applied physics
                query = f"{query_term} cat:cond-mat.mtrl-sci OR cat:physics.app-ph"
                search = arxiv.Search(
                    query=query,
                    max_results=1000,  # Limit to avoid timeout
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                # Count results
                doc_count = sum(1 for _ in search.results())
                doc_count = max(doc_count, 1)  # Avoid division by zero
                idf = math.log(N / doc_count)
                IDF_APPROX[term] = idf
                results.append({"Term": term, "Document Count": doc_count, "IDF": round(idf, 3)})
                logger.info(f"Term: {term}, Documents: {doc_count}, IDF: {idf:.3f}")
            except Exception as e:
                logger.error(f"Error for '{term}': {str(e)}")
                IDF_APPROX[term] = math.log(N / 10000)  # Fallback IDF
                results.append({"Term": term, "Document Count": "Error", "IDF": round(math.log(N / 10000), 3)})

            # Update progress
            progress = (i + 1) / len(terms)
            progress_bar.progress(progress)
            status_text.text(f"Processed {i + 1}/{len(terms)} terms: {term}")
            time.sleep(1)  # Respect arXiv rate limits (~1 request/second)

        # Save to JSON
        output_path = "idf_approx.json"
        try:
            with open(output_path, "w") as f:
                json.dump(IDF_APPROX, f, indent=4)
            logger.info(f"IDF_APPROX saved to {output_path}")
            st.success(f"IDF_APPROX saved to `{output_path}`")
        except Exception as e:
            logger.error(f"Error saving IDF_APPROX: {str(e)}")
            st.error(f"Error saving IDF_APPROX: {str(e)}")

        # Display results
        st.subheader("IDF Results")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # Download button
        with open(output_path, "r") as f:
            st.download_button(
                label="Download idf_approx.json",
                data=f,
                file_name="idf_approx.json",
                mime="application/json"
            )

# Instructions for integration
st.subheader("Integration Instructions")
st.markdown("""
1. **Download `idf_approx.json`** after computation.
2. Place it in your main app's directory (e.g., `/home/kindness/workstation/.../corpus_data/`).
3. Update your main app's code to load `idf_approx.json` instead of the hardcoded `IDF_APPROX`:
   ```python
   import json
   import math
   try:
       with open("idf_approx.json", "r") as f:
           IDF_APPROX = json.load(f)
       logger.info("Loaded arXiv-derived IDF_APPROX from idf_approx.json")
   except FileNotFoundError:
       logger.warning("idf_approx.json not found, using default IDF_APPROX")
       IDF_APPROX = {
           "study": math.log(1000 / 800), "analysis": math.log(1000 / 700),
           # ... (your original IDF_APPROX)
       }
   DEFAULT_IDF = math.log(100000 / 10000)  # Updated to match arXiv corpus
   ```
4. Run your main app (`streamlit run your_app.py`) and verify that TF-IDF scores reflect the new IDFs in visualizations.
""")

# Footer
st.markdown("---")
st.markdown("Understanding how common or rare is a word or phrase in additive manufacturing research.")

