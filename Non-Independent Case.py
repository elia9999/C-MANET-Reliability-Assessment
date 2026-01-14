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
R_ch_node = 50.0
tau = 1.0
T_MAX = 40.0

NUM_MC = 200   # Monte Carlo simulation runs


# ==================== Link survival probability ====================
def link_survival_probability(t, d, v, R):
    if t <= 0:
        return 1.0
    if d >= R or v <= 0:
        return 0.0
    return max(0.0, 1.0 - t / ((R - d) / v))


# ==================== Node class ====================
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

    def random_walk(self, shared_direction=None, rho=0.0, noise=0.12):
        if shared_direction is not None and random.random() < rho:
            self.direction = shared_direction + random.gauss(0, noise)
        step = v * 0.5
        self.x = min(max(self.x + step * math.cos(self.direction), 0), max_x)
        self.y = min(max(self.y + step * math.sin(self.direction), 0), max_y)

    def get_weighted_probability(self):
        return P * (self.energy / initial_energy)


# ==================== MANET ====================
class MANET:
    def __init__(self):
        self.nodes = [Node(i) for i in range(num_nodes)]
        self.cluster_heads = []

    def select_cluster_head(self):
        sorted_nodes = sorted(self.nodes, key=lambda n: n.get_weighted_probability(), reverse=True)
        k = max(1, int(num_nodes * P))
        self.cluster_heads = []
        for n in sorted_nodes[:k]:
            n.set_as_cluster_head()
            self.cluster_heads.append(n)

    def make_cluster(self):
        for n in self.nodes:
            n.cluster_head = None
        for n in self.nodes:
            if not n.is_cluster_head:
                best, best_idx = 0, None
                for i, ch in enumerate(self.cluster_heads):
                    d = ch.distance(n)
                    if d <= ch.communication_range:
                        score = (ch.energy / energy_threshold) * (1 - d / ch.communication_range)
                        if score > best:
                            best, best_idx = score, i
                n.cluster_head = best_idx

    def move(self, correlated=False, rho=0.8):
        if not correlated:
            for n in self.nodes:
                n.random_walk()
        else:
            dirs = {i: random.uniform(0, 2 * math.pi) for i in range(len(self.cluster_heads))}
            for n in self.nodes:
                cid = self.cluster_heads.index(n) if n.is_cluster_head else n.cluster_head
                n.random_walk(shared_direction=dirs.get(cid), rho=rho)


# ==================== Graph and path ====================
def build_graph(net):
    g = {n.node_id: [] for n in net.nodes}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            ni, nj = net.nodes[i], net.nodes[j]
            if ni.distance(nj) <= min(ni.communication_range, nj.communication_range):
                g[i].append(j)
                g[j].append(i)
    return g


def bfs(graph, net, s, t):
    if s == t:
        return [], []
    q = deque([(s, [])])
    seen = {s}
    while q:
        cur, dists = q.popleft()
        for nxt in graph[cur]:
            if nxt not in seen:
                dist = net.nodes[cur].distance(net.nodes[nxt])
                if nxt == t:
                    return dists + [dist], len(dists) + 1
                seen.add(nxt)
                q.append((nxt, dists + [dist]))
    return None, None


# ==================== Reliability ====================
def max_success_rate(net, s, t):
    g = build_graph(net)
    dists, hops = bfs(g, net, s, t)
    if dists is None:
        return 0.0
    best = 0.0
    for t_ in np.linspace(0, T_MAX, 150):
        p = 1.0
        for d in dists:
            p *= link_survival_probability(t_, d, v, R_ch_node)
        if t_ // tau < hops:
            p = 0.0
        best = max(best, p)
    return best


def evaluate(net):
    node2cluster = {}
    for n in net.nodes:
        node2cluster[n.node_id] = net.cluster_heads.index(n) if n.is_cluster_head else n.cluster_head

    intra, inter = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = max_success_rate(net, i, j)
            if node2cluster[i] == node2cluster[j]:
                intra.append(p)
            else:
                inter.append(p)

    return np.mean(intra), np.mean(inter)


# ==================== Monte Carlo main program ====================
if __name__ == "__main__":

    intra_ind, inter_ind = [], []
    intra_corr, inter_corr = [], []

    for mc in range(NUM_MC):
        net = MANET()
        net.select_cluster_head()
        net.make_cluster()
        net.move(correlated=False)
        Ri, Re = evaluate(net)
        intra_ind.append(Ri)
        inter_ind.append(Re)

        net = MANET()
        net.select_cluster_head()
        net.make_cluster()
        net.move(correlated=True, rho=0.8)
        Ri, Re = evaluate(net)
        intra_corr.append(Ri)
        inter_corr.append(Re)

    print("\n=== Monte Carlo Results (500 runs, mean values) ===")
    print(f"Independent mobility:")
    print(f"  Intra-cluster reliability = {np.mean(intra_ind):.4f}")
    print(f"  Inter-cluster reliability = {np.mean(inter_ind):.4f}")

    print(f"\nCorrelated (group) mobility:")
    print(f"  Intra-cluster reliability = {np.mean(intra_corr):.4f}")
    print(f"  Inter-cluster reliability = {np.mean(inter_corr):.4f}")

    print("\n✅ Monte Carlo simulation (500 runs) completed.")