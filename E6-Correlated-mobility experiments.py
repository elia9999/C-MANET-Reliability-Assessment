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
initial_energy = 30
energy_threshold = 10
k1_energy_factor = 0.5

v = 2.0
R_member = 30.0
R_ch_values = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

tau = 1.0
T_MAX = 40.0

NUM_MC = 500

# Correlated mobility parameters
rho = 0.8
sigma = 0.1

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
        self.theta = random.uniform(0, 2 * math.pi)
        self.energy = initial_energy

        self.is_cluster_head = False
        self.cluster_head = None
        self.communication_range = R_member
        self.r_ch_node = r_ch_node

    def set_as_cluster_head(self):
        self.is_cluster_head = True
        self.communication_range = self.r_ch_node

    def set_as_member(self):
        self.is_cluster_head = False
        self.cluster_head = None
        self.communication_range = R_member

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def move(self, theta_c=None, rho=0.0):
        theta_prev = self.theta

        if theta_c is not None:
            epsilon = random.gauss(0, sigma)
            self.theta = (1 - rho) * theta_prev + rho * theta_c + epsilon

        step = v * 0.5
        self.x = min(max(self.x + step * math.cos(self.theta), 0), max_x)
        self.y = min(max(self.y + step * math.sin(self.theta), 0), max_y)

    def get_weighted_probability(self):
        return P * (self.energy / initial_energy)

    def get_leader_prob(self, other):
        d = self.distance(other)
        if d > self.communication_range:
            return 0.0

        k_E = self.energy / energy_threshold - k1_energy_factor
        if k_E <= 0:
            return 0.0

        return k_E * (1 - d / self.communication_range)

# ==================== MANET ====================
class MANET:
    def __init__(self, r_ch_node):
        self.r_ch_node = r_ch_node
        self.nodes = [Node(i, r_ch_node) for i in range(num_nodes)]
        self.cluster_heads = []

    def reset_roles(self):
        self.cluster_heads = []
        for n in self.nodes:
            n.set_as_member()

    def select_cluster_head(self):
        self.reset_roles()
        sorted_nodes = sorted(self.nodes, key=lambda n: n.get_weighted_probability(), reverse=True)
        k = max(1, int(num_nodes * P))

        for n in sorted_nodes[:k]:
            n.set_as_cluster_head()
            self.cluster_heads.append(n)

    def make_cluster(self):
        for n in self.nodes:
            n.cluster_head = None

        for n in self.nodes:
            if not n.is_cluster_head:
                best_prob = 0.0
                best_idx = None

                for i, ch in enumerate(self.cluster_heads):
                    p = ch.get_leader_prob(n)
                    if p > best_prob:
                        best_prob = p
                        best_idx = i

                n.cluster_head = best_idx

    def move(self, correlated=False):
        if not correlated:
            for n in self.nodes:
                n.move()
        else:
            cluster_dirs = {
                i: random.uniform(0, 2 * math.pi)
                for i in range(len(self.cluster_heads))
            }

            for n in self.nodes:
                cid = self.cluster_heads.index(n) if n.is_cluster_head else n.cluster_head
                theta_c = cluster_dirs.get(cid, None)
                n.move(theta_c=theta_c, rho=rho)

# ==================== Graph ====================
def build_graph(net):
    graph = {n.node_id: [] for n in net.nodes}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            ni, nj = net.nodes[i], net.nodes[j]

            if ni.distance(nj) <= min(ni.communication_range, nj.communication_range):
                graph[i].append(j)
                graph[j].append(i)

    return graph

def bfs_path(graph, net, s, t):
    if s == t:
        return [s], []

    q = deque([(s, [s], [])])
    visited = {s}

    while q:
        cur, path, dists = q.popleft()

        for nxt in graph[cur]:
            if nxt not in visited:
                dist = net.nodes[cur].distance(net.nodes[nxt])
                new_path = path + [nxt]
                new_dists = dists + [dist]

                if nxt == t:
                    return new_path, new_dists

                visited.add(nxt)
                q.append((nxt, new_path, new_dists))

    return None, None

# ==================== Reliability ====================
def max_success(net, s, t):
    graph = build_graph(net)
    path, dists = bfs_path(graph, net, s, t)

    if path is None:
        return 0.0

    hops = len(path) - 1
    if hops == 0:
        return 1.0

    best = 0.0

    for t_ in np.linspace(0, T_MAX, 120):
        if int(t_ // tau) < hops:
            p = 0.0
        else:
            p = 1.0
            for d in dists:
                # Keep the original reliability calculation style:
                # use the cluster-head communication range as the effective range
                p *= link_survival_probability(t_, d, v, net.r_ch_node)

        best = max(best, p)

    return best

def evaluate(net):
    node2cluster = {}

    for n in net.nodes:
        node2cluster[n.node_id] = net.cluster_heads.index(n) if n.is_cluster_head else n.cluster_head

    intra, inter = [], []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = max_success(net, i, j)

            if node2cluster[i] == node2cluster[j]:
                intra.append(p)
            else:
                inter.append(p)

    return np.mean(intra), np.mean(inter)

# ==================== Statistics ====================
def calc_mean_ci95(values):
    arr = np.array(values, dtype=float)
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    se = std_val / math.sqrt(len(arr))
    ci95 = 1.96 * se
    return mean_val, ci95

# ==================== Monte Carlo for one R_ch_node ====================
def run_experiment_for_rch(r_ch_node):
    intra_ind, inter_ind = [], []
    intra_corr, inter_corr = [], []

    for mc in range(NUM_MC):
        # Independent mobility
        net = MANET(r_ch_node)
        net.select_cluster_head()
        net.make_cluster()
        net.move(correlated=False)
        Ri, Re = evaluate(net)
        intra_ind.append(Ri)
        inter_ind.append(Re)

        # Correlated mobility
        net = MANET(r_ch_node)
        net.select_cluster_head()
        net.make_cluster()
        net.move(correlated=True)
        Ri, Re = evaluate(net)
        intra_corr.append(Ri)
        inter_corr.append(Re)

    return {
        "R_ch_node": r_ch_node,
        "Intra_Independent_Mean": calc_mean_ci95(intra_ind)[0],
        "Intra_Independent_CI95": calc_mean_ci95(intra_ind)[1],
        "Inter_Independent_Mean": calc_mean_ci95(inter_ind)[0],
        "Inter_Independent_CI95": calc_mean_ci95(inter_ind)[1],
        "Intra_Correlated_Mean": calc_mean_ci95(intra_corr)[0],
        "Intra_Correlated_CI95": calc_mean_ci95(intra_corr)[1],
        "Inter_Correlated_Mean": calc_mean_ci95(inter_corr)[0],
        "Inter_Correlated_CI95": calc_mean_ci95(inter_corr)[1],
    }

# ==================== Main program ====================
if __name__ == "__main__":
    all_results = []

    for r_ch_node in R_ch_values:
        print(f"Running simulations for R_ch_node = {r_ch_node} ...")
        result = run_experiment_for_rch(r_ch_node)
        all_results.append(result)

    result_df = pd.DataFrame(all_results)

    output_file = "Rch_mobility_reliability_results.xlsx"
    result_df.to_excel(output_file, index=False)

    print(f"\nExcel saved: {output_file}")