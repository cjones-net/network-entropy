import re
from typing import Union

import networkx as nx
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf


class entropyGraph(nx.Graph):
    """
    Child of networkx Graph class, where additonal methods are added for
    measuring degree distributions and related statistics.

    Methods
    _______

    graph_from_file(path)

    reduce_and_relabel()

    degree_distribution()

    remaining_distribution()

    expected_degree()

    expected_degree_square()

    critical_point_theory()

    degree_groups()

    joint_distribution(cluster_adjust=False)

    mutual_information(cluster_adjust=False)
    """

    def __init__(self):
        "Intialises the entropyGraph class."

        super(entropyGraph, self).__init__()

    def graph_from_file(self, path):
        "Reads an edge file and converts it to a graph."

        with open(path) as input_file:
            lines = input_file.readlines()
        input_file.close()
        edge_list = [
            re.split("[,|\t| |\n]", l)[:2]
            for l in lines
            if not l.startswith(("%", "#"))
        ]
        self.add_edges_from(edge_list)

    def reduce_and_relabel(self):
        """
        Reduces a graph to its largest connected component and relabels its
        nodes.
        """

        largestComp = sorted(
            [list(c) for c in nx.connected_components(self)], key=len
        )[-1]

        self.remove_nodes_from(
            [node for node in self.nodes() if node not in largestComp]
        )
        mapping = dict(
            list(zip(sorted(self), list(range(self.number_of_nodes() + 1))))
        )
        nx.relabel_nodes(self, mapping, copy=False)

    def degree_distribution(self) -> list[float]:
        """
        Calculates the degree distribution of the network.

        Returns
        _______

        list: List of float values, corresponding to the probability of
        choosing a node with degree value given by the list index.
        """

        return [
            degree / self.number_of_nodes()
            for degree in nx.degree_histogram(self)
        ]

    def remaining_distribution(self) -> list[float]:
        """
        Calculates the remaining degree distribution of the network.

        Returns
        _______

        list: List of float values, corresponding to the probability of
        choosing an edge uniformly at random, and following the edge to a node
        with degree minus one given by the list index.
        """

        deg_dist = self.degree_distribution()
        return [
            (degree + 1) * deg_dist[degree + 1] / self.expected_degree()
            for degree in range(len(deg_dist) - 1)
        ]

    def expected_degree(self) -> float:
        """
        Calculates the average degree of nodes in the network.

        Returns
        _______

        float: Average degree value.
        """

        return 2 * self.number_of_edges() / self.number_of_nodes()

    def expected_degree_square(self) -> float:
        """
        Calculates the average squared degree value of nodes in the network.

        Returns
        _______

        float: Average squared degree value.
        """

        deg_dist = self.degree_distribution()
        return sum(
            (degree**2) * deg_dist[degree] for degree in range(len(deg_dist))
        )

    def critical_point_theory(self) -> float:
        """
        Calculate the theoretical critical fraction for a network.

        Returns
        _______

        float: Value corresponding to the ratio of nodes removed at
        random for which the network collapses.
        """

        return 1 - 1 / (
            self.expected_degree_square() / self.expected_degree() - 1
        )

    def degree_groups(self) -> dict[int, list[Union[str, int]]]:
        """
        Groups nodes with the same degree value together.

        Returns
        _______

        dict: Dictionary of node groups, with keys given by degree values.
        """

        deg_dist = self.degree_distribution()
        deg_groups_dict = {
            degree: []
            for degree in range(len(deg_dist))
            if deg_dist[degree] != 0
        }
        for node in self.nodes():
            deg_groups_dict[self.degree(node)].append(node)
        return deg_groups_dict

    def joint_distribution(self, cluster_adjust: bool = False) -> np.array:
        """
        Calculates the joint degree distribution for a network.

        Parameters
        __________

        cluster_adjust: Boolean value flagging whether to reduce degree values
        if nodes have common neighbours.

        Returns
        _______

        array: Probabilities of choosing connected node pairs with remaining
        degrees given by array indices.
        """

        # if adjusting for clustering, degree values of nodes are reduced
        # according to how many clusters they are in
        if cluster_adjust == True:
            deg_list = []
            for edge in self.edges():
                deg_adjust = len(
                    [
                        neigh
                        for neigh in nx.common_neighbors(
                            self, edge[0], edge[1]
                        )
                    ]
                )
                deg_list.append(
                    sorted(
                        [
                            self.degree(edge[0]) - deg_adjust,
                            self.degree(edge[1]) - deg_adjust,
                        ]
                    )
                )
        # if clustering is not adjusted for, degree values are recorded without
        # adjustment
        else:
            deg_list = [
                sorted([self.degree(edge[0]), self.degree(edge[1])])
                for edge in self.edges()
            ]
        # creates a matrix recording how many pairs of given degrees exist
        rem_dist = self.remaining_distribution()
        joint_deg_array = np.array([[0 for q in rem_dist] for r in rem_dist])
        for degree in deg_list:
            joint_deg_array[degree[0] - 1][degree[1] - 1] += 1
            joint_deg_array[degree[1] - 1][degree[0] - 1] += 1
        # normalises and returns the joint distribution
        return joint_deg_array / sum(sum(row) for row in joint_deg_array)

    def mutual_info(self, cluster_adjust) -> float:
        """
        Calculates the mutual information for a network.

        Parameters
        __________

        cluster_adjust: Boolean value flagging whether to reduce degree values
        if nodes have common neighbours.

        Returns
        _______

        float: Mutual information value for the network.
        """

        rem_dist = self.remaining_distribution()
        product_dist = np.array([q * np.array(rem_dist) for q in rem_dist])
        joint_dist = self.joint_distribution(cluster_adjust)
        return sum(
            sum(
                [
                    joint_dist[i][j]
                    * (np.log(joint_dist[i][j]) - np.log(product_dist[i][j]))
                    for i in range(len(joint_dist))
                    if product_dist[i][j] != 0 and joint_dist[i][j] != 0
                ]
            )
            for j in range(len(joint_dist))
        )


# performs a correlation preserving edge swap on a graph
def correlation_preserve_swap(graph, stayConnected=False, maxDepth=1000):
    # chooses a group of nodes according to their degree values
    chosenGroup = graph.degree_groups()[
        np.random.choice(
            range(len(graph.degree_distribution())),
            p=graph.degree_distribution(),
        )
    ]
    # initialises parameters for checking whether a successful swap has occurred, and how many failures have occurred
    successfulSwap = False
    depth = 0
    # attempts swaps until successful or until the maximum number of allowable failures occurs
    while successfulSwap == False:
        if depth < maxDepth:
            if len(chosenGroup) > 1:
                # if the degree group has more than two members, chooses two nodes u and v from the group
                # choses edges (u,x) and (v,y), removes these from the graph and adds (u,y) and (v,x)
                try:
                    nodeU, nodeV = np.random.choice(
                        chosenGroup, size=2, replace=False
                    )
                    nodeX = np.random.choice(
                        [
                            neigh
                            for neigh in graph[nodeU]
                            if neigh not in graph[nodeV]
                        ]
                    )
                    nodeY = np.random.choice(
                        [
                            neigh
                            for neigh in graph[nodeV]
                            if neigh not in graph[nodeU]
                        ]
                    )
                    graph.remove_edges_from([(nodeU, nodeX), (nodeV, nodeY)])
                    graph.add_edges_from([(nodeU, nodeY), (nodeV, nodeX)])
                    # ensures the graph remains connected after the swap, if this is required
                    if stayConnected == True:
                        if nx.is_connected(graph) == True:
                            successfulSwap = True
                        # if the graph is disconnected, reverses the swap and tries again
                        else:
                            graph.add_edges_from(
                                [(nodeU, nodeX), (nodeV, nodeY)]
                            )
                            graph.remove_edges_from(
                                [(nodeU, nodeY), (nodeV, nodeX)]
                            )
                            chosenGroup = graph.degree_groups()[
                                np.random.choice(
                                    range(len(graph.degree_distribution())),
                                    p=graph.degree_distribution(),
                                )
                            ]
                            depth += 1
                    else:
                        successfulSwap = True
                # if x and y cannot be chosen such that both (u,y) and (v,x) edges do not already exist, a new degree group is chosen
                except ValueError:
                    chosenGroup = graph.degree_groups()[
                        np.random.choice(
                            range(len(graph.degree_distribution())),
                            p=graph.degree_distribution(),
                        )
                    ]
                    depth += 1
            # if the chosen degree group has only one member, a new degree group is chosen
            else:
                chosenGroup = graph.degree_groups()[
                    np.random.choice(
                        range(len(graph.degree_distribution())),
                        p=graph.degree_distribution(),
                    )
                ]
                depth += 1
        # if too many failures occur, an exception is raised
        else:
            raise Exception(
                "Maximum recursion depth reached without finding suitable swap candidates."
            )


# calculates the critical fraction for random node removal via simulation
def sim_crit_frac(graph, targeting=False, criticalPercent=0.01):
    # if targeting by degree value, creates a target list in ascending degree order, randomised within degree groups
    if targeting == True:
        targetList = []
        for group in graph.degree_groups().values():
            targetList.extend(
                list(np.random.choice(group, len(group), replace=False))
            )
    # if not targeting by degree value, creates a target list in random order
    else:
        targetList = [n for n in graph.nodes()]
        np.random.shuffle(targetList)
    # intialises variables corresponding to an empty graph
    activeNodes = []
    largestComp = 0
    trees = {n: n for n in graph.nodes()}
    sizes = {n: 1 for n in graph.nodes()}
    # iteratively adds nodes to the empty graph until the largest component reaches a specified critical point
    while largestComp < round(graph.number_of_nodes() * criticalPercent):
        # selects the next node
        node = targetList[len(activeNodes)]
        # searches over all neighbours present in the graph to determine which component nodes belong to
        for neigh in graph.neighbors(node):
            if neigh in activeNodes:
                # finds the "roots" of the selected node and neighbour via a recursive search
                nodeRoot = trees[node]
                neighRoot = trees[neigh]
                while nodeRoot != trees[nodeRoot]:
                    trees[nodeRoot] = trees[trees[nodeRoot]]
                    nodeRoot = trees[nodeRoot]
                while neighRoot != trees[neighRoot]:
                    trees[neighRoot] = trees[trees[neighRoot]]
                    neighRoot = trees[neighRoot]
                # if the selected node and neighbour do not already have a common root, the labels and component sizes are updated
                if nodeRoot != neighRoot:
                    if sizes[nodeRoot] >= sizes[neighRoot]:
                        trees[neighRoot] = trees[nodeRoot]
                        sizes[nodeRoot] += sizes[neighRoot]
                        sizes[neighRoot] = 0
                    else:
                        trees[nodeRoot] = trees[neighRoot]
                        sizes[neighRoot] += sizes[nodeRoot]
                        sizes[nodeRoot] = 0
        # records the largest component size and updates the active nodes in the graph
        largestComp = max(sizes.values())
        activeNodes.append(node)
    return 1 - len(activeNodes) / graph.number_of_nodes()


# the "zed" function for the truncated normal distribution
def zed_func(mu, sigma):
    return 0.5 * (erf(mu / (sigma * (2**0.5))) + 1)


# the "phi" function for the truncated normal distribution
def phi_func(mu, sigma):
    return (2 * np.pi) ** -0.5 * np.exp(-((mu / sigma) ** 2) / 2)


# calculates ratio between first and second moments of the truncated normal distribution and then uses this to calculate Molloy-Reed critical fraction
def trunc_crit(mu, sigma):
    kappa = (
        mu**2
        + sigma**2
        + mu * sigma * (phi_func(mu, sigma) / zed_func(mu, sigma))
    ) / (mu + (sigma * (phi_func(mu, sigma) / zed_func(mu, sigma))))
    return 1 - float(1) / (kappa - 1)


# degree distribution entropy of the truncated normal distribution
def trunc_entropy(mu, sigma):
    truncEnt = np.log(
        ((2 * np.pi * np.exp(1)) ** 0.5) * sigma * zed_func(mu, sigma)
    ) - ((mu / (2 * sigma)) * (phi_func(mu, sigma) / zed_func(mu, sigma)))
    return truncEnt


# power law probability distribution
def power_law_distribution(alpha, minDegree, maxDegree=1000):
    norm = sum((nVal + minDegree) ** (-alpha) for nVal in range(maxDegree))
    return [(k ** (-alpha)) / norm for k in range(minDegree, maxDegree)]


# log normal probability distribution
def log_normal_distribution(mu, sigma, maxDegree=1000):
    rawDist = [
        np.exp(-((np.log(k) - mu) ** 2) / (2 * sigma**2))
        for k in range(1, maxDegree)
    ]
    return [0] + [p / sum(rawDist) for p in rawDist]


# given an expected degree and minimum degree value, finds the appropriate value of alpha for the power law distribution
def alpha_finder(minDegree, expectedDegree, maxDegree=1000):

    def alpha_min(
        alphaGuess,
        maxDegree=maxDegree,
        expectedDegree=expectedDegree,
        minDegree=minDegree,
    ):
        probDist = power_law_distribution(alphaGuess, minDegree, maxDegree)
        calcExpect = sum(
            k * p for k, p in zip(range(minDegree, maxDegree), probDist)
        )
        return np.abs(calcExpect - expectedDegree)

    alpha = minimize_scalar(alpha_min).x
    return alpha


# given an expected degree and sigma value, finds the appropriate value of mu for the log normal distribution
def mu_finder(sigma, expectedDegree, maxDegree=1000):

    def mu_min(
        muGuess,
        sigma=sigma,
        expectedDegree=expectedDegree,
        maxDegree=maxDegree,
    ):
        probDist = log_normal_distribution(muGuess, sigma, maxDegree)
        calcExpect = sum(k * p for k, p in zip(range(maxDegree), probDist))
        return np.abs(calcExpect - expectedDegree)

    mu = minimize_scalar(mu_min).x
    return mu
