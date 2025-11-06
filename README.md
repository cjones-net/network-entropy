# README

### For code used in the paper "Clarifying How Degree Entropies and Degree-Degree Correlations Relate to Network Robustness", C Jones and K Wiesner, Entropy 2022, 24(9), 1182; https://doi.org/10.3390/e24091182

This repository has been organised such that the data from each figure in the above paper may be independently generated and verified. The module "networkentropy" contains the functions that have been specially written by the authors for this paper.

The necessary packages which must be installed in order to run the scripts in this repository are given in `requirements.txt`. These packages may be installed with the command:

```
pip install -r requirements.txt
```

The `network_data` folder contains a range real world networks from http://konect.cc/ and http://networkrepository.com.

`real_nets_trunc_normal_calculations.py` generates Figure 1, which plots critical fraction values against degree distribution entropy for real networks and shows the theoretical bound.

`theory_entropies_crit_frac_calculations.py` generates Figure 2, plotting critical fraction values against entropies for power law and log normal distributions corresponding to networks of fixed average degree.

`mutual_info_calculations.py` generates Figures 4 & 6, plotting critical fraction values against edge swaps and adjusted mutual information against critical fraction values respectively for a real world network undergoing edge swaps.