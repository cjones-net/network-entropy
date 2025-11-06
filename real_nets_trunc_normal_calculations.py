import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import entropy

import networkentropy as nent

# specifies the directory of network data
directory = "network_data"
FIG_DIR = Path("~/network-entropy/figures").expanduser()
# initialises lists for degree distribution entropy and molloy-reed critical
# fraction
distEntropy, critFracs = [], []
# iterates over network data files, calculating degree distribution entropy and
# critical fraction values
for filename in os.listdir(directory)[:5]:
    graph = nent.entropyGraph()
    graph.graph_from_file(os.path.join(directory, filename))
    distEntropy.append(entropy(graph.degree_dist()))
    critFracs.append(graph.molloy_reed())

# sets the initial value for sigma and initialises lists for truncated normal data
sigma = 0.01
truncEntList = []
truncCritList = []
# iterates over values of sigma and calculates maximum entropy for a given critical fraction
while sigma < 100:
    mu = 0.84 * sigma
    truncEnt = nent.trunc_entropy(mu, sigma)
    truncCrit = nent.trunc_crit(mu, sigma)
    # filters out results for invalid critical fraction values
    if 1 >= truncCrit >= 0:
        truncEntList.append(truncEnt)
        truncCritList.append(truncCrit)
    sigma += 0.01

# plots data from both real networks and truncated normal distribution
fig = plt.figure()
ax = fig.add_subplot()
plt.xlim(0, 5.5)
plt.ylim(0, 1)
plt.plot(distEntropy, critFracs, "x", color="blue", label="Real Networks")
plt.plot(
    truncEntList,
    truncCritList,
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
