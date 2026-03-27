import random
import math
import numpy as np
import pandas as pd
from collections import deque

# ==================== Global parameters ====================
max_x = 100
max_y = 100
num_nodes = 50
P = 0.12
initial_energy = 50
energy_threshold = 10
k1_energy_factor = 0.5

v = 2.0
R_member = 30.0
R_ch_values = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

tau = 1.0
T_MAX = 40.0

# ==================== Interference parameters ====================
# Fixed interference level for the R_ch_node sensitivity study
INTERFERENCE_LEVEL = 1.0
BETA_I = 0.7

# H1_tau is used as the interference-dependent part of the per-hop transmission success rate
# consistent with P_Data^{i,j}(tau) = P(SNR >= theta)
def H1_tau(interference_level, beta=BETA_I):
    return float(np.exp(-beta * max(0.0, interference_level)))

# ==================== Monte Carlo parameters ====================
MC_RUNS = 200

# ==================== Link survival probability ====================
def link_survival_probability(t, d, v, R):
    if t <= 0:
        return 1.0
    if d >= R or v <= 0:
        return 0.0
    return max(0.0, 1.0 - t / ((R - d) / v))

# ==================== Node ====================
class Node:
    def __init__(self, node_id, r_ch_node):
        self.node_id = node_id
        self.x = random.uniform(0, max_x)
        self.y = random.uniform(0, max_y)
        self.direction = random.uniform(0, 2 * math.pi)
        self.energy = initial_energy
        self.is_cluster_head = False
        self.cluster_head = None
        self.communication_range = R_member
        self.r_ch_node = r_ch_node

    def set_as_cluster_head(self):
        self.is_cluster_head = True
        self.communication_range = self.r_ch_node

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def move(self):
        """
        Independent mobility only.
        """
        step = v * 0.5
        self.x = min(max(self.x + step * math.cos(self.direction), 0), max_x)
        self.y = min(max(self.y + step * math.sin(self.direction), 0), max_y)

# ==================== MANET ====================
class MANET:
    def __init__(self, r_ch_node):
        self.r_ch_node = r_ch_node
        self.nodes = [Node(i, r_ch_node) for i in range(num_nodes)]
        self.cluster_heads = []

    def select_cluster_head(self):
        k = max(1, int(num_nodes * P))
        self.cluster_heads = random.sample(self.nodes, k)
        for ch in self.cluster_heads:
            ch.set_as_cluster_head()

    def make_cluster(self):
        for node in self.nodes:
            if node.is_cluster_head:
                continue

            best = None
            best_d = float("inf")
            for ch in self.cluster_heads:
                d = node.distance(ch)
                if d < best_d:
                    best = ch
                    best_d = d

            node.cluster_head = self.cluster_heads.index(best)

    def move_all(self):
        for node in self.nodes:
            node.move()

# ==================== Graph and path ====================
def build_graph(net):
    graph = {n.node_id: [] for n in net.nodes}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            ni, nj = net.nodes[i], net.nodes[j]
            dij = ni.distance(nj)
            if dij <= ni.communication_range and dij <= nj.communication_range:
                graph[i].append(j)
                graph[j].append(i)

    return graph

def bfs(graph, net, s, t):
    """
    Return the list of hop distances from s to t using BFS shortest-hop routing.
    """
    if s == t:
        return []

    queue = deque([(s, [])])
    visited = {s}

    while queue:
        cur, dists = queue.popleft()
        for nb in graph[cur]:
            if nb in visited:
                continue

            visited.add(nb)
            new_dists = dists + [net.nodes[cur].distance(net.nodes[nb])]

            if nb == t:
                return new_dists

            queue.append((nb, new_dists))

    return None

# ==================== Reliability ====================
def single_hop_success_probability(t, d, interference_level, r_ch_node):
    """
    Per-hop transmission success probability.
    """
    p_link = link_survival_probability(t, d, v, r_ch_node)
    p_data = H1_tau(interference_level)
    return p_link * p_data

def path_success_probability(t, dists, interference_level, r_ch_node):
    """
    End-to-end path success probability as the product of per-hop success probabilities.
    """
    p = 1.0
    for d in dists:
        p *= single_hop_success_probability(t, d, interference_level, r_ch_node)
    return p

def transmission_success(t, dists, hops, interference_level, r_ch_node):
    """
    End-to-end transmission success rate under slot constraint.
    """
    if hops == 0:
        return 1.0
    if int(t // tau) < hops:
        return 0.0
    return path_success_probability(t, dists, interference_level, r_ch_node)

def max_success(src, tgt, net, interference_level):
    graph = build_graph(net)
    dists = bfs(graph, net, src, tgt)
    if dists is None:
        return 0.0

    hops = len(dists)
    ts = np.linspace(0, T_MAX, 200)
    return max(transmission_success(t, dists, hops, interference_level, net.r_ch_node) for t in ts)

def evaluate(net, interference_level):
    intra, inter = [], []

    node2cluster = {}
    for node in net.nodes:
        if node.is_cluster_head:
            node2cluster[node.node_id] = net.cluster_heads.index(node)
        else:
            node2cluster[node.node_id] = node.cluster_head

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pij = max_success(i, j, net, interference_level)
            if node2cluster[i] == node2cluster[j]:
                intra.append(pij)
            else:
                inter.append(pij)

    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    return intra_mean, inter_mean

# ==================== Statistics ====================
def calc_mean_ci95(values):
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = std_val / math.sqrt(len(arr)) if len(arr) > 0 else 0.0
    ci95 = 1.96 * se
    return mean_val, ci95

# ==================== Experiment for one R_ch_node ====================
def run_experiment_for_rch(r_ch_node, interference_level, mc_runs):
    intra_values, inter_values = [], []

    for _ in range(mc_runs):
        net = MANET(r_ch_node)
        net.select_cluster_head()
        net.make_cluster()
        net.move_all()

        r_in, r_out = evaluate(net, interference_level)
        intra_values.append(r_in)
        inter_values.append(r_out)

    return {
        "R_ch_node": r_ch_node,
        "Interference_Level": interference_level,
        "Intra_Mean": calc_mean_ci95(intra_values)[0],
        "Intra_CI95": calc_mean_ci95(intra_values)[1],
        "Inter_Mean": calc_mean_ci95(inter_values)[0],
        "Inter_CI95": calc_mean_ci95(inter_values)[1],
    }

# ==================== Main program ====================
if __name__ == "__main__":
    all_results = []

    print("\n[Monte Carlo study: R_ch_node sensitivity under fixed interference]")
    print(f"Fixed interference level I = {INTERFERENCE_LEVEL}")
    print(f"Monte Carlo runs per setting = {MC_RUNS}\n")

    for r_ch_node in R_ch_values:
        print(f"Running simulations for R_ch_node = {r_ch_node} ...")
        result = run_experiment_for_rch(
            r_ch_node=r_ch_node,
            interference_level=INTERFERENCE_LEVEL,
            mc_runs=MC_RUNS
        )
        all_results.append(result)

    result_df = pd.DataFrame(all_results)

    output_file = "Rch_proposed_structure_only.xlsx"
    result_df.to_excel(output_file, index=False)

    print(f"\nExcel saved: {output_file}")
    print("Done.")