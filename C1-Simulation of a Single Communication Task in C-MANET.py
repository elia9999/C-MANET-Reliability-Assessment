import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==================== Global parameters ====================
max_x = 100
max_y = 100
num_nodes = 50
P = 0.12  # cluster head proportion
initial_energy = 50
energy_threshold = 10
k1_energy_factor = 0.5

# Reliability parameters
v = 2.0  # mobility speed (m/s)
R_member = 30.0  # communication radius for ordinary member nodes (m)
R_ch_node = 60.0  # communication radius for cluster head nodes (m) — larger to improve backbone connectivity
tau = 1.0  # slot length (s)
T_MAX = 40.0  # maximum time (s)


# ==================== Link survival probability (distance-based) ====================
def link_survival_probability(t, d, v, R):
    """Link survival probability, simplified model based on equation (7) from the paper"""
    if t <= 0:
        return 1.0
    if d >= R or v <= 0:
        return 0.0
    max_t = (R - d) / v
    return max(0.0, 1.0 - t / max_t)


# ==================== Node class ====================
class Node:
    def __init__(self, node_id, max_x, max_y):
        self.node_id = node_id
        self.x = random.uniform(0, max_x)
        self.y = random.uniform(0, max_y)
        self.direction = random.uniform(0, 2 * math.pi)
        self.energy = initial_energy
        self.is_cluster_head = False
        self.cluster_head = None
        self.selected_round = -1
        # default to member communication range
        self.communication_range = R_member

    def set_as_cluster_head(self):
        """Mark this node as a cluster head and update its communication range"""
        self.is_cluster_head = True
        self.communication_range = R_ch_node

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def random_walk(self, step_size=1):
        step = v * 0.5
        self.x += step * math.cos(self.direction)
        self.y += step * math.sin(self.direction)
        self.x = max(0, min(max_x, self.x))
        self.y = max(0, min(max_y, self.y))

    def get_weighted_probability(self, round_num):
        if self.selected_round == round_num:
            return 0.0
        cycle_length = int(1 / P) if P > 0 else 1
        r_mod = round_num % cycle_length
        base_prob = P / (1 - P * r_mod) if r_mod < 1 / P else 0.0
        energy_ratio = self.energy / initial_energy
        return base_prob * energy_ratio

    def get_leader_prob(self, other):
        if self.distance(other) > self.communication_range:
            return 0.0
        k_E = self.energy / energy_threshold - k1_energy_factor
        if k_E <= 0:
            return 0.0
        d = self.distance(other)
        F_d = 1 - d / self.communication_range if d < self.communication_range else 0.0
        return k_E * F_d


# ==================== MANET network class ====================
class MANET:
    def __init__(self, num_nodes, max_x, max_y):
        self.num_nodes = num_nodes
        self.max_x = max_x
        self.max_y = max_y
        self.nodes = [Node(i, max_x, max_y) for i in range(num_nodes)]
        self.cluster_heads = []
        self.round = 0

    def select_cluster_head(self):
        sorted_nodes = sorted(
            self.nodes,
            key=lambda x: x.get_weighted_probability(self.round),
            reverse=True
        )
        k = max(1, int(self.num_nodes * P))
        self.cluster_heads = []
        for node in sorted_nodes[:k]:
            node.set_as_cluster_head()  # ← important: update to cluster-head communication range
            node.selected_round = self.round
            self.cluster_heads.append(node)

    def make_cluster(self):
        for node in self.nodes:
            node.cluster_head = None
        for node in self.nodes:
            if not node.is_cluster_head:
                best_ch_idx = None
                max_prob = 0
                for idx, ch in enumerate(self.cluster_heads):
                    prob = ch.get_leader_prob(node)
                    if prob > max_prob:
                        max_prob = prob
                        best_ch_idx = idx
                if best_ch_idx is not None:
                    node.cluster_head = best_ch_idx

    def random_walk_all(self):
        for node in self.nodes:
            node.random_walk()


# ==================== Graph construction and path search ====================
def build_communication_graph(manet):
    """
    Build an undirected graph according to each node's communication_range.
    An edge is created only if the bidirectional distance ≤ each node's communication radius (symmetric link).
    """
    graph = {node.node_id: [] for node in manet.nodes}
    nodes = manet.nodes
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            d = nodes[i].distance(nodes[j])
            # bidirectional reachability: i within j's range AND j within i's range
            if d <= nodes[i].communication_range and d <= nodes[j].communication_range:
                graph[nodes[i].node_id].append(nodes[j].node_id)
                graph[nodes[j].node_id].append(nodes[i].node_id)
    return graph


def bfs_shortest_path_with_distance(graph, manet, start, goal):
    """
    BFS to find the shortest-hop path, returning the node sequence and per-hop distances
    """
    if start == goal:
        return [start], []
    visited = set()
    queue = deque([(start, [start], [])])
    visited.add(start)
    node_dict = {n.node_id: n for n in manet.nodes}

    while queue:
        current, path, dists = queue.popleft()
        for neighbor in graph.get(current, []):
            n_current = node_dict[current]
            n_neighbor = node_dict[neighbor]
            d_hop = n_current.distance(n_neighbor)
            new_path = path + [neighbor]
            new_dists = dists + [d_hop]
            if neighbor == goal:
                return new_path, new_dists
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, new_path, new_dists))
    return None, None


# ==================== Obtain k1, k2, k3 (can be >1) ====================
def get_k_params_multi_hop(source, target, manet):
    """
    Strictly follow the C-MANET three-layer routing:
      1. Source member → its assigned cluster head (mandatory)
      2. Source cluster head ↔ Target cluster head (multi-hop backbone)
      3. Target cluster head → Target member
    Use the graph built with heterogeneous communication ranges.
    """
    graph = build_communication_graph(manet)  # ← use heterogeneous communication capabilities
    node_dict = {n.node_id: n for n in manet.nodes}
    result = {'k1': 0, 'k2': 0, 'k3': 0, 'valid': True}

    # --- Segment 1: source → its cluster head (mandatory) ---
    if source.is_cluster_head:
        src_ch_id = source.node_id
        result['path1'] = [source.node_id]
        result['distances1'] = []
        result['k1'] = 0
    else:
        if source.cluster_head is None:
            result['valid'] = False
            return None
        src_ch = manet.cluster_heads[source.cluster_head]  # ← use the assigned cluster head
        src_ch_id = src_ch.node_id
        path1, dists1 = bfs_shortest_path_with_distance(graph, manet, source.node_id, src_ch_id)
        if path1 is None:
            result['valid'] = False
            return None
        result['path1'] = path1
        result['distances1'] = dists1
        result['k1'] = len(path1) - 1

    # --- Segment 3: target cluster head → target (mandatory) ---
    if target.is_cluster_head:
        tgt_ch_id = target.node_id
        result['k2'] = 0
        result['path2'] = [target.node_id]
        result['distances2'] = []
    else:
        if target.cluster_head is None:
            result['valid'] = False
            return None
        tgt_ch = manet.cluster_heads[target.cluster_head]
        tgt_ch_id = tgt_ch.node_id
        path2_rev, dists2_rev = bfs_shortest_path_with_distance(graph, manet, tgt_ch_id, target.node_id)
        if path2_rev is None:
            result['valid'] = False
            return None
        result['path2'] = path2_rev
        result['distances2'] = dists2_rev
        result['k2'] = len(path2_rev) - 1

    # --- Segment 2: src_CH ↔ tgt_CH (multi-hop backbone) ---
    if src_ch_id == tgt_ch_id:
        result['k3'] = 0
        result['path3'] = [src_ch_id]
        result['distances3'] = []
    else:
        path3, dists3 = bfs_shortest_path_with_distance(graph, manet, src_ch_id, tgt_ch_id)
        if path3 is None:
            result['valid'] = False
            return None
        result['path3'] = path3
        result['distances3'] = dists3
        result['k3'] = len(path3) - 1

    return result


# ==================== Reliability calculation ====================
def path_stability_probability(t, k_info, v=v):
    """Path stability probability = product of link survival probabilities for each hop"""
    if k_info is None:
        return 0.0
    all_distances = (
            k_info['distances1'] +
            k_info['distances3'] +
            k_info['distances2']
    )
    prob = 1.0
    for i, d in enumerate(all_distances):
        # Use the smaller communication radius among the two endpoints for R (conservative estimate)
        # In practice one could use the sender or receiver R; using a uniform value simplifies trends
        R_used = R_ch_node  # or choose dynamically based on hop type
        prob *= link_survival_probability(t, d, v, R_used)
    return prob


def transmission_success_rate(t, k_info, tau=tau):
    """
    Transmission success rate = path stability probability × data transmission success probability
    Based on paper Section 3.c.ii:
      - P_data(t) = 1 iff number of available slots >= total hops
    """
    if k_info is None:
        return 0.0
    total_hops = k_info['k1'] + k_info['k3'] + k_info['k2']
    if total_hops == 0:
        return 1.0

    num_slots = int(t // tau)
    p_data = 1.0 if num_slots >= total_hops else 0.0
    p_stable = path_stability_probability(t, k_info)
    return p_stable * p_data


# ==================== Performance evaluation ====================
def evaluate_single_task(manet, src_id, tgt_id, time_points=None):
    if time_points is None:
        time_points = np.linspace(0, T_MAX, 100)

    src = next(n for n in manet.nodes if n.node_id == src_id)
    tgt = next(n for n in manet.nodes if n.node_id == tgt_id)
    k_info = get_k_params_multi_hop(src, tgt, manet)

    results = {'time': [], 'p_stable': [], 'p_success': []}
    for t in time_points:
        p_s = path_stability_probability(t, k_info)
        p_t = transmission_success_rate(t, k_info)
        results['time'].append(t)
        results['p_stable'].append(p_s)
        results['p_success'].append(p_t)
    return results, k_info


# ==================== Visualization ====================
def visualize_path(manet, k_info, title="C-MANET with Heterogeneous Communication Ranges"):
    if not k_info or not k_info.get('valid', False):
        print("No valid path to visualize.")
        return

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    node_dict = {n.node_id: n for n in manet.nodes}

    # draw all nodes
    for node in manet.nodes:
        color = 'red' if node.is_cluster_head else 'lightblue'
        marker = '^' if node.is_cluster_head else 'o'
        size = 120 if node.is_cluster_head else 80
        edge = 'k' if node.is_cluster_head else 'gray'
        plt.scatter(node.x, node.y, c=color, s=size, marker=marker,
                    edgecolors=edge, zorder=2)
        plt.text(node.x + 0.8, node.y + 0.8, str(node.node_id), fontsize=6)

    # draw communication range circles (only cluster heads to avoid clutter)
    for node in manet.nodes:
        if node.is_cluster_head:
            circle = plt.Circle((node.x, node.y), node.communication_range,
                                color='red', alpha=0.08)
            ax.add_patch(circle)
        else:
            circle = plt.Circle((node.x, node.y), node.communication_range,
                                color='blue', alpha=0.03)
            ax.add_patch(circle)

    # draw path
    full_path = []
    if k_info['k1'] > 0:
        full_path.extend(k_info['path1'][:-1])
    full_path.extend(k_info['path3'])
    if k_info['k2'] > 0:
        full_path.extend(k_info['path2'][1:])

    if len(full_path) > 1:
        xs = [node_dict[nid].x for nid in full_path]
        ys = [node_dict[nid].y for nid in full_path]
        plt.plot(xs, ys, 'b--', linewidth=2.5, label='Communication Path')
        # annotate hops
        for i in range(len(xs) - 1):
            mid_x = (xs[i] + xs[i + 1]) / 2
            mid_y = (ys[i] + ys[i + 1]) / 2
            plt.text(mid_x, mid_y, str(i + 1), color='darkblue', fontweight='bold',
                     ha='center', va='center', fontsize=9, zorder=5)

    # highlight source and target
    src_id = full_path[0]
    tgt_id = full_path[-1]
    plt.scatter(node_dict[src_id].x, node_dict[src_id].y, c='gold', s=200,
                edgecolors='red', linewidth=2, zorder=6, label='Source')
    plt.scatter(node_dict[tgt_id].x, node_dict[tgt_id].y, c='lime', s=200,
                edgecolors='green', linewidth=2, zorder=6, label='Target')

    plt.title(title, fontsize=13)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.xlim(-5, max_x + 5)
    plt.ylim(-5, max_y + 5)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ==================== Main program ====================
if __name__ == "__main__":
    print("Initializing C-MANET with heterogeneous communication ranges...")
    net = MANET(num_nodes=num_nodes, max_x=max_x, max_y=max_y)
    net.random_walk_all()
    net.select_cluster_head()
    net.make_cluster()

    src_id = random.choice([n.node_id for n in net.nodes])
    tgt_id = random.choice([n.node_id for n in net.nodes])
    print(f"Selected task: Node {src_id} --> Node {tgt_id}")

    src_node = next(n for n in net.nodes if n.node_id == src_id)
    tgt_node = next(n for n in net.nodes if n.node_id == tgt_id)
    k_info = get_k_params_multi_hop(src_node, tgt_node, net)

    if k_info is None:
        print("❌ No valid path found between source and target.")
    else:
        print(f"✅ Path found:")
        print(f"  k1 (src→src_CH) = {k_info['k1']}")
        print(f"  k3 (CH↔CH)      = {k_info['k3']}")
        print(f"  k2 (tgt_CH→tgt) = {k_info['k2']}")
        print(f"  Total hops      = {k_info['k1'] + k_info['k3'] + k_info['k2']}")

        results, _ = evaluate_single_task(net, src_id, tgt_id)
        visualize_path(net, k_info)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(results['time'], results['p_stable'], 'g-', linewidth=2)
        plt.title("Path Stability Probability")
        plt.xlabel("Time (s)")
        plt.ylabel("P_stable(t)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(results['time'], results['p_success'], 'r-', linewidth=2)
        plt.title("Transmission Success Rate")
        plt.xlabel("Time (s)")
        plt.ylabel("P_success(t)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        t_final = results['time'][-1]
        p_final = results['p_success'][-1]
        print(f"\nAt t = {t_final:.1f}s, Transmission Success Rate = {p_final:.4f}")