import random
import math
import numpy as np
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
R_ch_node = 55.0
tau = 1.0
T_MAX = 40.0

# ==================== Interference parameters ====================
BETA_I = 0.7

def H1_tau(interference_level, beta=BETA_I):
    return float(np.exp(-beta * max(0.0, interference_level)))

# ==================== Link survival probability ====================
def link_survival_probability(t, d, v, R):
    if t <= 0:
        return 1.0
    if d >= R or v <= 0:
        return 0.0
    return max(0.0, 1.0 - t / ((R - d) / v))

# ==================== Node ====================
class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.x = random.uniform(0, max_x)
        self.y = random.uniform(0, max_y)
        self.direction = random.uniform(0, 2 * math.pi)
        self.energy = initial_energy
        self.is_cluster_head = False
        self.cluster_head = None
        self.communication_range = R_member

    def set_as_cluster_head(self):
        self.is_cluster_head = True
        self.communication_range = R_ch_node

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def random_walk(self):
        step = v * 0.5
        self.x = min(max(self.x + step * math.cos(self.direction), 0), max_x)
        self.y = min(max(self.y + step * math.sin(self.direction), 0), max_y)

# ==================== MANET ====================
class MANET:
    def __init__(self):
        self.nodes = [Node(i) for i in range(num_nodes)]
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

    def random_walk_all(self):
        for node in self.nodes:
            node.random_walk()

# ==================== Graph and path ====================
def build_graph(net):
    graph = {n.node_id: [] for n in net.nodes}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            ni, nj = net.nodes[i], net.nodes[j]
            if ni.distance(nj) <= ni.communication_range and ni.distance(nj) <= nj.communication_range:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def bfs(graph, net, s, t):
    if s == t:
        return [], []
    queue = deque([(s, [])])
    visited = {s}
    while queue:
        cur, dists = queue.popleft()
        for nb in graph[cur]:
            if nb in visited:
                continue
            visited.add(nb)
            nd = dists + [net.nodes[cur].distance(net.nodes[nb])]
            if nb == t:
                return nd
            queue.append((nb, nd))
    return None

# ==================== Reliability ====================
def path_stability(t, dists):
    p = 1.0
    for d in dists:
        p *= link_survival_probability(t, d, v, R_ch_node)
    return p


def transmission_success(t, dists, hops, I):
    if hops == 0:
        return 1.0
    if int(t // tau) < hops:
        return 0.0
    return path_stability(t, dists) * (H1_tau(I) ** hops)


def max_success(src, tgt, net, I):
    graph = build_graph(net)
    dists = bfs(graph, net, src, tgt)
    if dists is None:
        return 0.0
    hops = len(dists)
    ts = np.linspace(0, T_MAX, 200)
    return max(transmission_success(t, dists, hops, I) for t in ts)


def evaluate(net, I):
    intra, inter = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pi = max_success(i, j, net, I)
            if net.nodes[i].cluster_head == net.nodes[j].cluster_head:
                intra.append(pi)
            else:
                inter.append(pi)
    return np.mean(intra), np.mean(inter)

# ==================== Monte Carlo main program ====================
if __name__ == "__main__":
    MC_RUNS = 200
    interference_levels = [0.0, 1.0,2.0]

    print("\n[Monte Carlo Sensitivity Study: 200 runs averaged]")
    print(f"{'I':>6} | {'H1(τ)':>10} | {'R_intra(avg)':>14} | {'R_inter(avg)':>14}")
    print("-" * 80)

    for I in interference_levels:
        Rin, Rie = [], []
        for _ in range(MC_RUNS):
            net = MANET()
            net.random_walk_all()
            net.select_cluster_head()
            net.make_cluster()
            r_in, r_ie = evaluate(net, I)
            Rin.append(r_in)
            Rie.append(r_ie)

        print(f"{I:6.2f} | {H1_tau(I):10.4f} | {np.mean(Rin):14.4f} | {np.mean(Rie):14.4f}")

    print("-" * 80)
    print("✅ 200-run Monte Carlo interference sensitivity analysis completed.")