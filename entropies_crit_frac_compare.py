from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

import networkentropy as nent

FIG_DIR = Path("~/network-entropy/figures").expanduser()

max_degree = 1000
expected_degree = 10

power_law_degree_entropies = []
power_law_remaining_entropies = []
power_crit_fractions = []
min_degree = 1
# iterates power law distributions over minimum degree values
while min_degree < 10:
    # finds value for alpha
    alpha = nent.alpha_finder(min_degree, expected_degree)
    # generates power law degree distribution and measures degree distribution
    # entropy
    power_law_prob_dist = nent.power_law_distribution(alpha, min_degree)
    power_law_degree_entropies.append(entropy(power_law_prob_dist))
    # finds first and second moments and calculates Molloy-Reed critical
    # fraction
    power_law_expect_deg = sum(
        k * p
        for k, p in zip(range(min_degree, max_degree), power_law_prob_dist)
    )
    power_law_expect_deg_square = sum(
        (k**2) * p
        for k, p in zip(range(min_degree, max_degree), power_law_prob_dist)
    )
    power_crit_fractions.append(
        1 - 1 / (power_law_expect_deg_square / power_law_expect_deg - 1)
    )
    # measures power law remaining degree entropy
    power_law_remaining_entropies.append(
        entropy(
            [
                k * p / power_law_expect_deg
                for k, p in zip(
                    range(min_degree, max_degree), power_law_prob_dist
                )
            ]
        )
    )
    # increases minimum degree value for next iteration
    min_degree += 1

log_norm_degree_entropies = []
log_norm_remaining_entropies = []
log_norm_crit_fractions = []
sigma = 0.2
# iterates numerical log normal distributions over sigma values
while sigma < 3:
    # finds value for mu
    mu = nent.mu_finder(sigma, expected_degree)
    # generates log normal degree distribution and measures degree distribution entropy
    log_norm_prob_dist = nent.log_normal_distribution(mu, sigma)
    log_norm_degree_entropies.append(entropy(log_norm_prob_dist))
    # finds first and second moments and calculates Molloy-Reed critical fraction
    log_norm_expect_deg = sum(
        k * p for k, p in zip(range(max_degree), log_norm_prob_dist)
    )
    log_norm_expect_deg_square = sum(
        (k**2) * p for k, p in zip(range(max_degree), log_norm_prob_dist)
    )
    log_norm_crit_fractions.append(
        1 - 1 / (log_norm_expect_deg_square / log_norm_expect_deg - 1)
    )
    # measures log normal remaining degree entropy
    log_norm_remaining_entropies.append(
        entropy(
            [
                k * p / log_norm_expect_deg
                for k, p in zip(range(max_degree), log_norm_prob_dist)
            ]
        )
    )
    # increases sigma value for next iteration
    sigma += 0.2

theory_degree_entropies = []
theory_remaining_entropies = []
theory_crit_fractions = []
theory_sigma = 0.01
# iterates theoretical log normal distributions over sigma values
while theory_sigma < 3:
    # calculates entropy values and Molloy Reed critical fraction
    theory_degree_entropies.append(
        0.5 * (1 - theory_sigma**2)
        + np.log(expected_degree * theory_sigma * (2 * np.pi) ** 0.5)
    )
    theory_remaining_entropies.append(
        0.5 * (1 + theory_sigma**2)
        + np.log(expected_degree * theory_sigma * (2 * np.pi) ** 0.5)
    )
    theory_crit_fractions.append(
        1 - 1 / (expected_degree * np.exp(theory_sigma**2) - 1)
    )
    # increases sigma value for next iteration
    theory_sigma += 0.01

# plots degree distribution entropy against Molloy-Reed critical fraction
plt.figure()
plt.xlim(1, 3.5)
plt.ylim(0.88, 1)
plt.plot(
    power_law_degree_entropies,
    power_crit_fractions,
    "^",
    color="blue",
    label="Power-Law\n(numerical)",
)
plt.plot(
    log_norm_degree_entropies,
    log_norm_crit_fractions,
    "s",
    color="red",
    label="Log-Normal\n(numerical)",
)
plt.plot(
    theory_degree_entropies,
    theory_crit_fractions,
    linestyle="dashed",
    color="black",
    label="Log-Normal\n(theoretical)",
)
plt.xlabel(r"$H(p)$", fontdict={"fontsize": 16})
plt.ylabel(r"$f_c$", fontdict={"fontsize": 16}, rotation=0)
plt.legend(loc="upper left")
plt.savefig(FIG_DIR / "critical_frac_against_degree_dist_entropies.png")

# plots remaining degree entropy against Molloy-Reed critical fraction
fig = plt.figure()
ax = fig.add_subplot()
plt.xlim(1, 7)
plt.ylim(0.88, 1)
plt.plot(
    power_law_remaining_entropies,
    power_crit_fractions,
    "^",
    color="blue",
    label="Power-Law\n(numerical)",
)
plt.plot(
    log_norm_remaining_entropies,
    log_norm_crit_fractions,
    "s",
    color="red",
    label="Log-Normal\n(numerical)",
)
plt.plot(
    theory_remaining_entropies,
    theory_crit_fractions,
    linestyle="dashed",
    color="black",
    label="Log-Normal\n(theoretical)",
)
ax.set_xlabel(r"$H(q)$", fontdict={"fontsize": 16})
ax.set_ylabel(r"$f_c$", fontdict={"fontsize": 16}, rotation=0)
plt.legend(loc="upper left")
plt.savefig(FIG_DIR / "critical_frac_against_remaining_dist_entropies.png")
