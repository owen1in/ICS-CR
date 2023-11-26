# ICS-CR

This repository contains replication files for the master thesis: "Careless Responding Detection Using Invariant Coordinate Selection (ICS)".

Every file starts with a category number from 1 to 3.

1.
sim_data_generation.Rmd creates and saves simulated questionnaire data which are used in sim_methods.Rmd where the methods are applied.

2.
Full_data_survey.csv contains real-world data in which a portion of participants are instructed to exhibit careless responding. ipip_data_prep.Rmd loads this data and performs data preprocessing. After saving the preprocessed data, the methods are applied in the ipip_methods.Rmd script.

3.
example.Rmd generates the examples given in the Methodology section of the thesis along with the outputs.
inspect.Rmd is used to generate the IC plots in the Results section as well as the corresponding heat maps of the corresponding eigenvector matrices.
outputs.py is used to generate all other figures and tables in the thesis.
