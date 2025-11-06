import networkentropy as nent
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm


FIG_DIR = Path("~/network-entropy/figures").expanduser()

# generates the graph from an edge list file
graph = nent.entropyGraph()
graph.graph_from_file(
    Path("~/network-entropy/network_data/fb-pages-tvshow.edges").expanduser()
)
graph.reduce_and_relabel()
# sets up the initial parameters and intervals for the swapping procedure
swaps = [0, 100, 200, 400, 800, 1600, 3200]
iterations = 100
success_count = 0
# initialises the lists for recording critical fraction and mutual information
# values
(
    random_crit_frac_averages,
    random_crit_frac_errors,
    targeted_crit_frac_averages,
    targeted_crit_frac_errors,
    standard_mutual_information,
    cluster_adjust_mutual_information,
) = ([], [], [], [], [], [])

# iterates over edge swaps until reaching a cutoff number of swaps
for success_count in tqdm(range(max(swaps) + 1)):
    # at set intervals, records mutual information and critical fraction values
    if success_count in swaps:
        # informs the user when measurements are being taken
        print("Successful swaps = " + str(success_count))
        standard_mutual_information.append(
            graph.mutual_information(cluster_adjust=False)
        )
        cluster_adjust_mutual_information.append(
            graph.mutual_information(cluster_adjust=True)
        )
        # simulates the critical fraction measurement multiple times in order
        # to obtain an average
        random_crit_fracs = [
            nent.critical_point_simulation(graph) for i in range(iterations)
        ]
        targeted_crit_fracs = [
            nent.critical_point_simulation(graph, targeting=True)
            for i in range(iterations)
        ]
        random_crit_frac_averages.append(np.average(random_crit_fracs))
        random_crit_frac_errors.append(np.std(random_crit_fracs))
        targeted_crit_frac_averages.append(np.average(targeted_crit_fracs))
        targeted_crit_frac_errors.append(np.std(targeted_crit_fracs))
    # performs a correlation preserving swap on the graph
    nent.correlation_preserve_swap(graph, stay_connected=True)

print(
    "Check that all standard mutual information values are the same: "
    f"{all(standard_mutual_information)}."
)

# plots random critical fraction against swaps
plt.figure()
plt.errorbar(
    y=random_crit_frac_averages,
    x=swaps,
    yerr=random_crit_frac_errors,
    fmt="s",
    capsize=2,
    color="green",
    label="Random Failure",
)
plt.errorbar(
    y=random_crit_frac_averages, x=swaps, color="green", linestyle="dashed"
)
plt.xlim(-100, 3500)
plt.ylim(
    min(random_crit_frac_averages) - max(random_crit_frac_errors) - 0.01,
    max(random_crit_frac_averages) + max(random_crit_frac_errors) + 0.01,
)
plt.legend()
plt.ylabel(r"$f_c$", fontdict={"fontsize": 16}, rotation=0)
plt.xlabel("Number of Edge Swaps", fontdict={"fontsize": 12})
plt.savefig(FIG_DIR / "random_critical_frac_against_edge_swaps.png")

# plots mutual information with clustering against random critical fraction
plt.figure()
plt.errorbar(
    x=random_crit_frac_averages,
    y=cluster_adjust_mutual_information,
    xerr=random_crit_frac_errors,
    fmt="s",
    capsize=2,
    color="green",
    label="Random Failure",
)
plt.errorbar(
    x=random_crit_frac_averages,
    y=cluster_adjust_mutual_information,
    color="green",
    linestyle="dashed",
)
plt.xlim(
    min(random_crit_frac_averages) - max(random_crit_frac_errors) - 0.01,
    max(random_crit_frac_averages) + max(random_crit_frac_errors) + 0.01,
)
plt.ylim(
    min(cluster_adjust_mutual_information) - 0.1,
    max(cluster_adjust_mutual_information) + 0.1,
)
plt.legend()
plt.ylabel("Mutual Information with Clustering", fontdict={"fontsize": 12})
plt.xlabel(r"$f_c$", fontdict={"fontsize": 16})
plt.savefig(FIG_DIR / "mutual_information_against_random_critical_frac.png")

# plots targeted critical fraction against swaps
plt.figure()
plt.errorbar(
    y=targeted_crit_frac_averages,
    x=swaps,
    yerr=targeted_crit_frac_errors,
    fmt="o",
    capsize=2,
    color="purple",
    label="Targeted Attack",
)
plt.errorbar(
    y=targeted_crit_frac_averages, x=swaps, color="purple", linestyle="dashed"
)
plt.xlim(-100, 3500)
plt.ylim(
    min(targeted_crit_frac_averages) - max(targeted_crit_frac_errors) - 0.01,
    max(targeted_crit_frac_averages) + max(targeted_crit_frac_errors) + 0.01,
)
plt.legend()
plt.ylabel(r"$f_c$", fontdict={"fontsize": 12}, rotation=0)
plt.xlabel("Number of Edge Swaps", fontdict={"fontsize": 12})
plt.savefig(FIG_DIR / "targeted_critical_frac_against_edge_swaps.png")

# plots mutual information with clustering against targeted critical fraction
plt.figure()
plt.errorbar(
    x=targeted_crit_frac_averages,
    y=cluster_adjust_mutual_information,
    xerr=targeted_crit_frac_errors,
    fmt="o",
    capsize=2,
    color="purple",
    label="Targeted Attack",
)
plt.errorbar(
    x=targeted_crit_frac_averages,
    y=cluster_adjust_mutual_information,
    color="purple",
    linestyle="dashed",
)
plt.ylim(
    min(cluster_adjust_mutual_information) - 0.1,
    max(cluster_adjust_mutual_information) + 0.1,
)
plt.xlim(
    min(targeted_crit_frac_averages) - max(targeted_crit_frac_errors) - 0.01,
    max(targeted_crit_frac_averages) + max(targeted_crit_frac_errors) + 0.01,
)
plt.legend()
plt.ylabel("Mutual Information with Clustering", fontdict={"fontsize": 12})
plt.xlabel(r"$f_c$", fontdict={"fontsize": 16})
plt.savefig(FIG_DIR / "mutual_information_against_targeted_critical_frac.png")
