import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

import networkentropy as nent

FIG_DIR = Path("~/network-entropy/figures").expanduser()
NET_DIR = Path("~/network-entropy/network_data").expanduser()

# initialises lists for degree distribution entropy and molloy-reed critical
# fraction
distribution_entropies, critical_fractions = [], []

# iterates over network data files, calculating degree distribution entropy and
# critical fraction values
for filename in tqdm(os.listdir(NET_DIR)):
    graph = nent.entropyGraph()
    graph.graph_from_file(os.path.join(NET_DIR, filename))
    distribution_entropies.append(entropy(graph.degree_distribution()))
    critical_fractions.append(graph.critical_point_theory())

# initialises lists for truncated normal  data
trunc_normal_entropies, truncated_normal_crit_fracs = [], []
for sigma in tqdm(np.arange(0.01, 100, 0.01)):
    mu = 0.84 * sigma
    trunc_entropy = nent.trunc_entropy(mu, sigma)
    trunc_crit = nent.trunc_critical_point(mu, sigma)
    # filters out results for invalid critical fraction values
    if 1 >= trunc_crit >= 0:
        trunc_normal_entropies.append(trunc_entropy)
        truncated_normal_crit_fracs.append(trunc_crit)

# plots data from both real networks and truncated normal distribution
fig = plt.figure()
ax = fig.add_subplot()
plt.xlim(0, 5.5)
plt.ylim(0, 1)
plt.plot(
    distribution_entropies,
    critical_fractions,
    "x",
    color="blue",
    label="Real Networks",
)
plt.plot(
    trunc_normal_entropies,
    truncated_normal_crit_fracs,
    linestyle="dashed",
    color="black",
    label="Truncated Normal",
)
ax.set_xlabel(r"$H(p)$", fontdict={"fontsize": 12})
ax.set_ylabel(r"$f_c$", fontdict={"fontsize": 12}, rotation=0)
plt.legend()
plt.savefig(
    FIG_DIR / "critical_frac_against_real_networks_degree_entropies.png"
)
