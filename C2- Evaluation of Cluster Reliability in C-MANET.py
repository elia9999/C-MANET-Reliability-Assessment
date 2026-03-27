import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==================== Global Parameters ====================
max_x = 100
max_y = 100
num_nodes = 50
P = 0.12  # Cluster head ratio
initial_energy = 50
energy_threshold = 10
k1_energy_factor = 0.5

# Reliability parameters
v = 2.0  # Movement speed (m/s)
R_member = 30.0  # Communication range of ordinary member nodes (m)
R_ch_node = 60.0  # Communication range of cluster head nodes (m)
tau = 1.0  # Time slot length (s)
T_MAX = 40.0  # Maximum time (s)


# ==================== Link Survival Probability (Distance-based) ====================
def link_survival_probability(t, d, v, R):
    """Link survival probability, based on simplified model from paper Eq.(7)"""
    if t <= 0:
        return 1.0
    if d >= R or v <= 0:
        return 0.0
    max_t = (R - d) / v
    return max(0.0, 1.0 - t / max_t)


# ==================== Node Class ====================
class Node:
    def __init__(self, node_id, max_x, max_y):
        self.node_id = node_id
        self.x = random.uniform(0, max_x)
        self.y = random.uniform(0, max_y)
        self.direction = random.uniform(0, 2 * math.pi)
        self.energy = initial_energy
        self.is_cluster_head = False
        self.cluster_head = None
        self.selected_round = -1  # Default to member communication range
        self.communication_range = R_member

    def set_as_cluster_head(self):
        """Set this node as cluster head and update communication range"""
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


# ==================== MANET Network Class ====================
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
            node.set_as_cluster_head()
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


# ==================== Graph Construction and Path Search ====================
def build_communication_graph(manet):
    graph = {node.node_id: [] for node in manet.nodes}
    nodes = manet.nodes
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            d = nodes[i].distance(nodes[j])
            if d <= nodes[i].communication_range and d <= nodes[j].communication_range:
                graph[nodes[i].node_id].append(nodes[j].node_id)
                graph[nodes[j].node_id].append(nodes[i].node_id)
    return graph


def bfs_shortest_path_with_distance(graph, manet, start, goal):
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


# ==================== Get k1, k2, k3 (can be >1) ====================
def get_k_params_multi_hop(source, target, manet):
    graph = build_communication_graph(manet)
    node_dict = {n.node_id: n for n in manet.nodes}
    result = {'k1': 0, 'k2': 0, 'k3': 0, 'valid': True}

    # --- Segment 1: source -> its cluster head ---
    if source.is_cluster_head:
        src_ch_id = source.node_id
        result['path1'] = [source.node_id]
        result['distances1'] = []
        result['k1'] = 0
    else:
        if source.cluster_head is None:
            result['valid'] = False
            return None
        src_ch = manet.cluster_heads[source.cluster_head]
        src_ch_id = src_ch.node_id
        path1, dists1 = bfs_shortest_path_with_distance(graph, manet, source.node_id, src_ch_id)
        if path1 is None:
            result['valid'] = False
            return None
        result['path1'] = path1
        result['distances1'] = dists1
        result['k1'] = len(path1) - 1

    # --- Segment 3: target cluster head -> target ---
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

    # --- Segment 2: src_CH <-> tgt_CH ---
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


# ==================== Reliability Calculation ====================
def path_stability_probability(t, k_info, v=v):
    if k_info is None:
        return 0.0
    all_distances = (
            k_info['distances1'] +
            k_info['distances3'] +
            k_info['distances2']
    )
    prob = 1.0
    for d in all_distances:
        R_used = R_ch_node
        prob *= link_survival_probability(t, d, v, R_used)
    return prob


def transmission_success_rate(t, k_info, tau=tau):
    if k_info is None:
        return 0.0
    total_hops = k_info['k1'] + k_info['k3'] + k_info['k2']
    if total_hops == 0:
        return 1.0
    num_slots = int(t // tau)
    p_data = 1.0 if num_slots >= total_hops else 0.0
    p_stable = path_stability_probability(t, k_info)
    return p_stable * p_data


# ==================== Performance Evaluation ====================
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
    for node in manet.nodes:
        color = 'red' if node.is_cluster_head else 'lightblue'
        marker = '^' if node.is_cluster_head else 'o'
        size = 120 if node.is_cluster_head else 80
        edge = 'k' if node.is_cluster_head else 'gray'
        plt.scatter(node.x, node.y, c=color, s=size, marker=marker, edgecolors=edge, zorder=2)
        plt.text(node.x + 0.8, node.y + 0.8, str(node.node_id), fontsize=6)
    for node in manet.nodes:
        if node.is_cluster_head:
            circle = plt.Circle((node.x, node.y), node.communication_range, color='red', alpha=0.08)
            ax.add_patch(circle)
        else:
            circle = plt.Circle((node.x, node.y), node.communication_range, color='blue', alpha=0.03)
            ax.add_patch(circle)
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
        for i in range(len(xs) - 1):
            mid_x = (xs[i] + xs[i + 1]) / 2
            mid_y = (ys[i] + ys[i + 1]) / 2
            plt.text(mid_x, mid_y, str(i + 1), color='darkblue', fontweight='bold',
                     ha='center', va='center', fontsize=9, zorder=5)
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


# ==================== Calculate Maximum Transmission Success Rate for a Single Pair ====================
def get_max_success_rate_for_pair(src_node, tgt_node, manet, t_max=T_MAX, num_steps=200):
    """
    Calculate the maximum transmission success rate for a source-target node pair within [0, t_max] time.
    """
    k_info = get_k_params_multi_hop(src_node, tgt_node, manet)
    if k_info is None:
        return 0.0

    time_points = np.linspace(0, t_max, num_steps)
    max_p = 0.0
    for t in time_points:
        p = transmission_success_rate(t, k_info)
        if p > max_p:
            max_p = p
        if max_p >= 1.0:  # Early termination
            break
    return max_p


# ==================== Exhaustive Evaluation of Average Maximum Intra/Inter-cluster Reliability ====================
def evaluate_intra_inter_by_max_success(manet, t_max=T_MAX, num_time_steps=200):
    """
    Enumerate all unordered node pairs, calculate the maximum P_success(t) for each pair,
    group by whether they belong to the same cluster, and return the average maximum reliability for both categories.
    """
    node_dict = {node.node_id: node for node in manet.nodes}
    node_ids = sorted(node_dict.keys())
    n = len(node_ids)

    # Build node_id -> cluster_id mapping
    node_to_cluster = {}
    for node in manet.nodes:
        if node.is_cluster_head:
            try:
                cid = manet.cluster_heads.index(node)
            except ValueError:
                cid = -1
        else:
            cid = node.cluster_head
        node_to_cluster[node.node_id] = cid

    intra_list = []
    inter_list = []

    for i in range(n):
        for j in range(i + 1, n):
            s_id = node_ids[i]
            t_id = node_ids[j]
            s_node = node_dict[s_id]
            t_node = node_dict[t_id]

            c1 = node_to_cluster.get(s_id, None)
            c2 = node_to_cluster.get(t_id, None)

            if c1 is None or c2 is None:
                continue  # Skip unassigned nodes

            max_p = get_max_success_rate_for_pair(s_node, t_node, manet, t_max=t_max, num_steps=num_time_steps)

            if c1 == c2:
                intra_list.append(max_p)
            else:
                inter_list.append(max_p)

    R_intra = np.mean(intra_list) if intra_list else 0.0
    R_inter = np.mean(inter_list) if inter_list else 0.0

    return R_intra, R_inter, len(intra_list), len(inter_list)


# ==================== Main Program ====================
def run_single_simulation():
    """
    Execute a complete C-MANET reliability evaluation.
    Returns: (R_intra, R_inter)
    """
    net = MANET(num_nodes=num_nodes, max_x=max_x, max_y=max_y)
    net.random_walk_all()
    net.select_cluster_head()
    net.make_cluster()

    # Build node_id -> cluster_id mapping (for verifying same cluster)
    node_to_cluster = {}
    for node in net.nodes:
        if node.is_cluster_head:
            try:
                cid = net.cluster_heads.index(node)
            except ValueError:
                cid = -1
        else:
            cid = node.cluster_head
        node_to_cluster[node.node_id] = cid

    # Verify if there are any unassigned nodes (should not happen in theory, but for safety)
    unassigned = sum(1 for node in net.nodes if node_to_cluster[node.node_id] is None)
    if unassigned > 0:
        print(f"Warning: {unassigned} nodes unassigned in a simulation run.")

    # Evaluate intra/inter-cluster maximum reliability
    R_intra, R_inter, N_intra, N_inter = evaluate_intra_inter_by_max_success(
        net, t_max=T_MAX, num_time_steps=200
    )
    return R_intra, R_inter


# ==================== Main Program: Monte Carlo Simulation ====================
if __name__ == "__main__":
    import time
    NUM_SIMULATIONS = 500  # Number of Monte Carlo simulations
    print(f"🚀 Starting Monte Carlo Simulation ({NUM_SIMULATIONS} runs)...\n")

    intra_reliabilities = []
    inter_reliabilities = []

    start_time = time.time()
    for k in range(1, NUM_SIMULATIONS + 1):
        R_intra, R_inter = run_single_simulation()
        intra_reliabilities.append(R_intra)
        inter_reliabilities.append(R_inter)

        # Print progress every 50 runs
        if k % 50 == 0 or k == NUM_SIMULATIONS:
            elapsed = time.time() - start_time
            print(f"[{k}/{NUM_SIMULATIONS}] Done | "
                  f"Intra: {R_intra:.4f}, Inter: {R_inter:.4f} | "
                  f"Elapsed: {elapsed:.1f}s")

    # --- Statistical Results ---
    mean_intra = np.mean(intra_reliabilities)
    std_intra = np.std(intra_reliabilities, ddof=1)  # Sample standard deviation
    var_intra = np.var(intra_reliabilities, ddof=1)

    mean_inter = np.mean(inter_reliabilities)
    std_inter = np.std(inter_reliabilities, ddof=1)
    var_inter = np.var(inter_reliabilities, ddof=1)

    # --- Print Final Statistics ---
    print("\n" + "="*60)
    print("✅ Monte Carlo Simulation Results (500 runs):")
    print(f"🔹 Intra-cluster Reliability:")
    print(f"    Mean = {mean_intra:.4f}, Std = {std_intra:.4f}, Variance = {var_intra:.6f}")
    print(f"🔹 Inter-cluster Reliability:")
    print(f"    Mean = {mean_inter:.4f}, Std = {std_inter:.4f}, Variance = {var_inter:.6f}")
    print(f"🔹 Ratio (Mean Intra / Mean Inter) = {mean_intra / mean_inter:.2f}")
    print("="*60)

    # --- Plot: Reliability vs. Simulation Run Index ---
    plt.figure(figsize=(12, 6))
    sim_indices = np.arange(1, NUM_SIMULATIONS + 1)

    plt.plot(sim_indices, intra_reliabilities, 'r-', alpha=0.7, linewidth=1, label='Intra-cluster Reliability')
    plt.plot(sim_indices, inter_reliabilities, 'b-', alpha=0.7, linewidth=1, label='Inter-cluster Reliability')

    # Add mean reference lines
    plt.axhline(mean_intra, color='red', linestyle='--', linewidth=2, label=f'Mean Intra = {mean_intra:.4f}')
    plt.axhline(mean_inter, color='blue', linestyle='--', linewidth=2, label=f'Mean Inter = {mean_inter:.4f}')

    plt.title('Monte Carlo Simulation: Cluster Reliability vs. Run Index', fontsize=14)
    plt.xlabel('Simulation Run Index', fontsize=12)
    plt.ylabel('Average Maximum Reliability', fontsize=12)
    plt.xlim(1, NUM_SIMULATIONS)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # --- Optional: Plot Histograms ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(intra_reliabilities, bins=30, color='salmon', alpha=0.7, edgecolor='k')
    plt.axvline(mean_intra, color='red', linestyle='--', linewidth=2)
    plt.title(f'Intra-cluster Reliability\nMean={mean_intra:.4f}, Std={std_intra:.4f}')
    plt.xlabel('Reliability')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(inter_reliabilities, bins=30, color='skyblue', alpha=0.7, edgecolor='k')
    plt.axvline(mean_inter, color='blue', linestyle='--', linewidth=2)
    plt.title(f'Inter-cluster Reliability\nMean={mean_inter:.4f}, Std={std_inter:.4f}')
    plt.xlabel('Reliability')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print("\n✅ Monte Carlo simulation and visualization completed.")
