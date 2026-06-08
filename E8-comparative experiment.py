import random
import math
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import ttest_ind

# ==================== Global parameters ====================
max_x = 100
max_y = 100
num_nodes = 50
P = 0.28                  # cluster-head ratio
initial_energy = 50
energy_threshold = 10
k1_energy_factor = 0.5

# Reliability parameters
v = 2.0
R_member = 30.0
R_ch_values = [ 70,75, 80,85, 90]
tau = 1.0
T_MAX = 40.0

NUM_SIMULATIONS = 500

# ==================== Link survival probability ====================
def link_survival_probability(t, d, v, R):
    if t <= 0:
        return 1.0
    if d >= R or v <= 0:
        return 0.0
    max_t = (R - d) / v
    if max_t <= 0:
        return 0.0
    return max(0.0, 1.0 - t / max_t)

# ==================== Node ====================
class Node:
    def __init__(self, node_id, max_x, max_y, r_ch_node):
        self.node_id = node_id
        self.x = random.uniform(0, max_x)
        self.y = random.uniform(0, max_y)
        self.direction = random.uniform(0, 2 * math.pi)
        self.energy = initial_energy
        self.is_cluster_head = False
        self.cluster_head = None
        self.selected_round = -1
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

    def random_walk(self):
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

# ==================== MANET ====================
class MANET:
    def __init__(self, num_nodes, max_x, max_y, r_ch_node):
        self.num_nodes = num_nodes
        self.max_x = max_x
        self.max_y = max_y
        self.r_ch_node = r_ch_node
        self.nodes = [Node(i, max_x, max_y, r_ch_node) for i in range(num_nodes)]
        self.cluster_heads = []
        self.round = 0

    def random_walk_all(self):
        for node in self.nodes:
            node.random_walk()

    def select_cluster_head(self):
        for node in self.nodes:
            node.is_cluster_head = False
            node.communication_range = R_member

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
                max_prob = 0.0
                for idx, ch in enumerate(self.cluster_heads):
                    prob = ch.get_leader_prob(node)
                    if prob > max_prob:
                        max_prob = prob
                        best_ch_idx = idx
                if best_ch_idx is not None:
                    node.cluster_head = best_ch_idx

    def set_all_nodes_flat_ranges(self, R=R_member):
        self.cluster_heads = []
        for node in self.nodes:
            node.is_cluster_head = False
            node.cluster_head = None
            node.communication_range = R

# ==================== Graph and path search ====================
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
    visited = set([start])
    queue = deque([(start, [start], [])])
    node_dict = {n.node_id: n for n in manet.nodes}

    while queue:
        cur, path, dists = queue.popleft()
        for nb in graph.get(cur, []):
            n_cur = node_dict[cur]
            n_nb = node_dict[nb]
            d_hop = n_cur.distance(n_nb)
            new_path = path + [nb]
            new_dists = dists + [d_hop]
            if nb == goal:
                return new_path, new_dists
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, new_path, new_dists))
    return None, None

# ==================== Clustered routing ====================
def get_k_params_multi_hop(source, target, manet):
    graph = build_communication_graph(manet)
    result = {'k1': 0, 'k2': 0, 'k3': 0, 'valid': True}

    # segment 1: source -> source CH
    if source.is_cluster_head:
        src_ch_id = source.node_id
        result['path1'] = [source.node_id]
        result['distances1'] = []
        result['k1'] = 0
    else:
        if source.cluster_head is None:
            return None
        src_ch = manet.cluster_heads[source.cluster_head]
        src_ch_id = src_ch.node_id
        path1, dists1 = bfs_shortest_path_with_distance(graph, manet, source.node_id, src_ch_id)
        if path1 is None:
            return None
        result['path1'] = path1
        result['distances1'] = dists1
        result['k1'] = len(path1) - 1

    # segment 3: target CH -> target
    if target.is_cluster_head:
        tgt_ch_id = target.node_id
        result['path2'] = [target.node_id]
        result['distances2'] = []
        result['k2'] = 0
    else:
        if target.cluster_head is None:
            return None
        tgt_ch = manet.cluster_heads[target.cluster_head]
        tgt_ch_id = tgt_ch.node_id
        path2, dists2 = bfs_shortest_path_with_distance(graph, manet, tgt_ch_id, target.node_id)
        if path2 is None:
            return None
        result['path2'] = path2
        result['distances2'] = dists2
        result['k2'] = len(path2) - 1

    # segment 2: source CH <-> target CH
    if src_ch_id == tgt_ch_id:
        result['path3'] = [src_ch_id]
        result['distances3'] = []
        result['k3'] = 0
    else:
        path3, dists3 = bfs_shortest_path_with_distance(graph, manet, src_ch_id, tgt_ch_id)
        if path3 is None:
            return None
        result['path3'] = path3
        result['distances3'] = dists3
        result['k3'] = len(path3) - 1

    return result

# ==================== Flat routing ====================
def get_k_params_flat(source, target, manet):
    graph = build_communication_graph(manet)
    path, dists = bfs_shortest_path_with_distance(graph, manet, source.node_id, target.node_id)
    if path is None:
        return None
    return {
        'k1': len(path) - 1,
        'k2': 0,
        'k3': 0,
        'valid': True,
        'path1': path,
        'distances1': dists,
        'path2': [target.node_id],
        'distances2': [],
        'path3': [source.node_id],
        'distances3': []
    }

# ==================== Reliability ====================
def _extract_full_path(k_info):
    if k_info is None or not k_info.get('valid', False):
        return None

    full_path = []
    if k_info.get('k1', 0) > 0:
        full_path.extend(k_info['path1'][:-1])
    else:
        full_path.extend(k_info['path1'])

    if 'path3' in k_info and len(k_info['path3']) > 0:
        if len(full_path) > 0 and full_path[-1] == k_info['path3'][0]:
            full_path.extend(k_info['path3'][1:])
        else:
            full_path.extend(k_info['path3'])

    if k_info.get('k2', 0) > 0:
        if len(full_path) > 0 and full_path[-1] == k_info['path2'][0]:
            full_path.extend(k_info['path2'][1:])
        else:
            full_path.extend(k_info['path2'])

    return full_path if len(full_path) > 0 else None

def path_stability_probability(t, k_info, manet):
    if k_info is None:
        return 0.0

    full_path = _extract_full_path(k_info)
    if full_path is None or len(full_path) == 1:
        return 1.0

    node_dict = {n.node_id: n for n in manet.nodes}
    prob = 1.0
    for i in range(len(full_path) - 1):
        u = node_dict[full_path[i]]
        w = node_dict[full_path[i + 1]]
        d = u.distance(w)
        R_used = min(u.communication_range, w.communication_range)
        prob *= link_survival_probability(t, d, v, R_used)

    return prob

def transmission_success_rate(t, k_info, manet):
    if k_info is None:
        return 0.0

    total_hops = k_info['k1'] + k_info['k3'] + k_info['k2']
    if total_hops == 0:
        return 1.0

    num_slots = int(t // tau)
    p_data = 1.0 if num_slots >= total_hops else 0.0
    p_stable = path_stability_probability(t, k_info, manet)
    return p_stable * p_data

def get_max_success_rate_for_pair(src_node, tgt_node, manet, k_info_func, t_max=T_MAX, num_steps=200):
    k_info = k_info_func(src_node, tgt_node, manet)
    if k_info is None:
        return 0.0

    time_points = np.linspace(0, t_max, num_steps)
    max_p = 0.0
    for t in time_points:
        p = transmission_success_rate(t, k_info, manet)
        if p > max_p:
            max_p = p
        if max_p >= 1.0:
            break
    return max_p

# ==================== Exhaustive evaluations ====================
def evaluate_intra_inter_by_max_success(manet, t_max=T_MAX, num_time_steps=200):
    node_dict = {node.node_id: node for node in manet.nodes}
    node_ids = sorted(node_dict.keys())
    n = len(node_ids)

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
            s_node = node_dict[node_ids[i]]
            t_node = node_dict[node_ids[j]]

            c1 = node_to_cluster.get(s_node.node_id, None)
            c2 = node_to_cluster.get(t_node.node_id, None)
            if c1 is None or c2 is None:
                continue

            max_p = get_max_success_rate_for_pair(
                s_node, t_node, manet,
                k_info_func=get_k_params_multi_hop,
                t_max=t_max, num_steps=num_time_steps
            )

            if c1 == c2:
                intra_list.append(max_p)
            else:
                inter_list.append(max_p)

    R_intra = np.mean(intra_list) if intra_list else 0.0
    R_inter = np.mean(inter_list) if inter_list else 0.0
    return R_intra, R_inter

def evaluate_flat_all_pairs_by_max_success(manet, t_max=T_MAX, num_time_steps=200):
    node_dict = {node.node_id: node for node in manet.nodes}
    node_ids = sorted(node_dict.keys())
    n = len(node_ids)

    all_list = []
    for i in range(n):
        for j in range(i + 1, n):
            s = node_dict[node_ids[i]]
            t = node_dict[node_ids[j]]
            max_p = get_max_success_rate_for_pair(
                s, t, manet,
                k_info_func=get_k_params_flat,
                t_max=t_max, num_steps=num_time_steps
            )
            all_list.append(max_p)

    return np.mean(all_list) if all_list else 0.0

# ==================== One simulation under one R_ch ====================
def run_single_simulation_compare(r_ch_node):
    net = MANET(num_nodes=num_nodes, max_x=max_x, max_y=max_y, r_ch_node=r_ch_node)
    net.random_walk_all()

    # Clustered C-MANET
    net.select_cluster_head()
    net.make_cluster()
    R_intra, R_inter = evaluate_intra_inter_by_max_success(
        net, t_max=T_MAX, num_time_steps=200
    )

    # Flat baseline on the same realization
    net.set_all_nodes_flat_ranges(R=R_member)
    R_flat = evaluate_flat_all_pairs_by_max_success(
        net, t_max=T_MAX, num_time_steps=200
    )

    return R_intra, R_inter, R_flat

# ==================== Statistics ====================
def mean_ci95(values):
    arr = np.array(values, dtype=float)
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    se = std_val / math.sqrt(len(arr))
    ci95 = 1.96 * se
    return mean_val, ci95

def ci_non_overlap(mean1, ci1, mean2, ci2):
    low1, high1 = mean1 - ci1, mean1 + ci1
    low2, high2 = mean2 - ci2, mean2 + ci2
    return (high1 < low2) or (high2 < low1)

# ==================== Main program ====================
if __name__ == "__main__":
    all_results = []

    print(f"🚀 Starting Monte Carlo Simulation ({NUM_SIMULATIONS} runs for each R_ch)...\n")

    for r_ch in R_ch_values:
        print(f"Running simulations for R_ch_node = {r_ch} ...")

        intra_reliabilities = []
        inter_reliabilities = []
        flat_reliabilities = []

        for _ in range(NUM_SIMULATIONS):
            R_intra, R_inter, R_flat = run_single_simulation_compare(r_ch)
            intra_reliabilities.append(R_intra)
            inter_reliabilities.append(R_inter)
            flat_reliabilities.append(R_flat)

        mean_intra, ci_intra = mean_ci95(intra_reliabilities)
        mean_inter, ci_inter = mean_ci95(inter_reliabilities)
        mean_flat, ci_flat = mean_ci95(flat_reliabilities)

        # Welch t-tests
        t_intra_vs_flat, p_intra_vs_flat = ttest_ind(intra_reliabilities, flat_reliabilities, equal_var=False)
        t_inter_vs_flat, p_inter_vs_flat = ttest_ind(inter_reliabilities, flat_reliabilities, equal_var=False)

        all_results.append({
            "R_ch_node": r_ch,

            "C-MANET_Intra_Mean": mean_intra,
            "C-MANET_Intra_CI95": ci_intra,

            "C-MANET_Inter_Mean": mean_inter,
            "C-MANET_Inter_CI95": ci_inter,

            "Flat_Mean": mean_flat,
            "Flat_CI95": ci_flat,

            "Intra_vs_Flat_t_stat": t_intra_vs_flat,
            "Intra_vs_Flat_p_value": p_intra_vs_flat,
            "Intra_vs_Flat_CI_NonOverlap": ci_non_overlap(mean_intra, ci_intra, mean_flat, ci_flat),

            "Inter_vs_Flat_t_stat": t_inter_vs_flat,
            "Inter_vs_Flat_p_value": p_inter_vs_flat,
            "Inter_vs_Flat_CI_NonOverlap": ci_non_overlap(mean_inter, ci_inter, mean_flat, ci_flat),
        })

    result_df = pd.DataFrame(all_results)

    output_file = "Rch_clustered_vs_flat_mean_ci95_significance.xlsx"
    result_df.to_excel(output_file, index=False)

    print(f"\n✅ Excel saved: {output_file}")
    print("Done.")