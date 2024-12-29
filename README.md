# thermal_history_and_structural_effects(2024)

#FEM
...................

#MD
...................
# Visualization of heat source for P = 250.W and 350 W. 
[![Gaussian Heat Source via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gaussianheatsource.streamlit.app/)
# postprocessing
a. the temperature-time-vn.py constructs the temporal variation of Temperature of given points  at different vertical depths e.g. 0, 15 and 30 micrometer from the surface of a powder bed, when processed by laser in two scans (forward scan and return scan) at two input power values i.e. 250 W and 350 W. The scan speed for laser with P = 250 W is 1200 mm/s whereas that for laser with higher power is 800 mm/s. 

b. it might be necessary to merge the individual csv files downloaded at different timesteps size to a single csv file consisting the column of "t(ms)". the merging of csv files can be performed via csvmergertoolkit made available at https://github.com/anilkunwar/csvmergertoolkit
