# Thermal history of laser processing and structural effects in AlMgSiZr multicomponent alloys (2024)

# FEM (Continuum scale modeling)
...................

# Meanings and Inferences from Words and Phrases of Additive Manufacturing (Attention Mechanism and Statistical Approach)

wordcloud and networks of words
[![meaningtowords](https://img.shields.io/badge/WordPhraseGraphs-streamlit-red)](https://visualizationofwordsphrases.streamlit.app/)

wordcloud, networks and radar charts for frequency and relevance metrices
[![meaningtowords](https://img.shields.io/badge/AdvancedWordPhraseGraphs-streamlit-red)](https://advancedgraphswordsphrases.streamlit.app/)

further efforts to enhance the visualization of word and phrase networks
- Basic model
  [![meaningtowords](https://img.shields.io/badge/basicwordnetwork-streamlit-red)](https://basicvisualizationofwordphrasegraphs.streamlit.app/)
- enhanced model 
[![meaningtowords](https://img.shields.io/badge/enhancedwordnetwork-streamlit-red)](https://enhancedvisualizationofwordphrasegraphs.streamlit.app/)

quantifying the words and phrases from data-driven study

computation of the inverse document frequency of a single "word" or a "single phrase" using the data information available in arxiv database
[![meaningtowords](https://img.shields.io/badge/computeidf-streamlit-red)](https://singlewordphraseidfcomputation.streamlit.app/)

(display of tf, idf and tf-idf) computation of the inverse document frequency of a single "word" or a "single phrase" using the data information available in arxiv database
[![meaningtowords](https://img.shields.io/badge/computetfidf-streamlit-red)](https://advancedsingletermtfidfcomputation.streamlit.app/)

(display of tf, idf ,tf-idf and queried documents count) computation of the inverse document frequency of a single "word" or a "single phrase" using the data information available in arxiv database
[![meaningtowords](https://img.shields.io/badge/computetfidffromndoc-streamlit-red)](https://https://advancedsingletermtfidfcomputefromdocuments.streamlit.app/)
 

graphs of words and phrases using information of tf-idf from documents of arxiv database
[![meaningtowords](https://img.shields.io/badge/visualizewordsphrases-streamlit-red)](https://datadrivenstudyofwordsphrases.streamlit.app/)


# Visualization of heat source for P = 250.0 W and 350.0 W. 
[![Gaussian Heat Source via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gaussianheatsourcemodel.streamlit.app/)

# Visualization of T-dependence of Hollomon parameters. 
[![Hollomon parameters via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hollomonparameters.streamlit.app/)

# MD (Atomistic simulations)
...................

a. a postprocessor to compute the unstable stacking fault energy and ideal shear strength from the GSFE-displacement data

# postprocessing
a. the temperature-time-vn.py constructs the temporal variation of Temperature of given points  at different vertical depths e.g. 0, 15 and 30 micrometer from the surface of a powder bed, when processed by laser in two scans (forward scan and return scan) at two input power values i.e. 250 W and 350 W. The scan speed for laser with P = 250 W is 1200 mm/s whereas that for laser with higher power is 800 mm/s. 

b. it might be necessary to merge the individual csv files downloaded at different timesteps size to a single csv file consisting the column of "t(ms)". the merging of csv files can be performed via csvmergertoolkit made available at https://github.com/anilkunwar/csvmergertoolkit
