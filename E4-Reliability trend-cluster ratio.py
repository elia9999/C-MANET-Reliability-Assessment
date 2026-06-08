import random
import math
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm

# ==================== 全局参数 ====================
max_x = 100
max_y = 100
num_nodes = 50

P_values = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28]

initial_energy = 30
energy_threshold = 10
k1_energy_factor = 0.5

v = 2.0
R_member = 30.0
R_ch_node = 60.0
tau = 1.0
T_MAX = 40.0

NUM_MC = 100


# ==================== 链路生存概率 ====================
def link_survival_probability(t, d, v, R):
    if t <= 0:
        return 1.0
    if d >= R or v <= 0:
        return 0.0

    max_t = (R - d) / v
    if max_t <= 0:
        return 0.0

    return max(0.0, 1.0 - t / max_t)


# ==================== 节点类 ====================
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
        self.communication_range = R_member

    def set_as_cluster_head(self):
        self.is_cluster_head = True
        self.communication_range = R_ch_node

    def reset_as_member(self):
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

    def get_weighted_probability(self, round_num, P):
        if self.selected_round == round_num:
            return 0.0

        cycle_length = int(1 / P) if P > 0 else 1
        r_mod = round_num % cycle_length
        base_prob = P / (1 - P * r_mod) if r_mod < 1 / P else 0.0
        energy_ratio = self.energy / initial_energy

        return base_prob * energy_ratio

    def get_leader_prob(self, other):
        d = self.distance(other)

        if d > self.communication_range:
            return 0.0

        k_E = self.energy / energy_threshold - k1_energy_factor

        if k_E <= 0:
            return 0.0

        F_d = 1 - d / self.communication_range if d < self.communication_range else 0.0

        return k_E * F_d


# ==================== MANET 网络类 ====================
class MANET:
    def __init__(self, num_nodes, max_x, max_y, P):
        self.num_nodes = num_nodes
        self.max_x = max_x
        self.max_y = max_y
        self.P = P
        self.nodes = [Node(i, max_x, max_y) for i in range(num_nodes)]
        self.cluster_heads = []
        self.round = 0

    def random_walk_all(self):
        for node in self.nodes:
            node.random_walk()

    def select_cluster_head(self):
        for node in self.nodes:
            node.reset_as_member()

        sorted_nodes = sorted(
            self.nodes,
            key=lambda x: x.get_weighted_probability(self.round, self.P),
            reverse=True
        )

        k = max(1, int(self.num_nodes * self.P))
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


# ==================== 图构建与路径搜索 ====================
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

    visited = {start}
    queue = deque([(start, [start], [])])
    node_dict = {n.node_id: n for n in manet.nodes}

    while queue:
        current, path, dists = queue.popleft()

        for neighbor in graph.get(current, []):
            if neighbor in visited:
                continue

            d_hop = node_dict[current].distance(node_dict[neighbor])
            new_path = path + [neighbor]
            new_dists = dists + [d_hop]

            if neighbor == goal:
                return new_path, new_dists

            visited.add(neighbor)
            queue.append((neighbor, new_path, new_dists))

    return None, None


# ==================== C-MANET 三段路由 ====================
def get_k_params_multi_hop(source, target, manet, graph):
    result = {
        "k1": 0,
        "k2": 0,
        "k3": 0,
        "valid": True
    }

    if source.is_cluster_head:
        src_ch_id = source.node_id
        result["path1"] = [source.node_id]
        result["distances1"] = []
        result["k1"] = 0
    else:
        if source.cluster_head is None:
            return None

        src_ch = manet.cluster_heads[source.cluster_head]
        src_ch_id = src_ch.node_id

        path1, dists1 = bfs_shortest_path_with_distance(
            graph,
            manet,
            source.node_id,
            src_ch_id
        )

        if path1 is None:
            return None

        result["path1"] = path1
        result["distances1"] = dists1
        result["k1"] = len(path1) - 1

    if target.is_cluster_head:
        tgt_ch_id = target.node_id
        result["path2"] = [target.node_id]
        result["distances2"] = []
        result["k2"] = 0
    else:
        if target.cluster_head is None:
            return None

        tgt_ch = manet.cluster_heads[target.cluster_head]
        tgt_ch_id = tgt_ch.node_id

        path2, dists2 = bfs_shortest_path_with_distance(
            graph,
            manet,
            tgt_ch_id,
            target.node_id
        )

        if path2 is None:
            return None

        result["path2"] = path2
        result["distances2"] = dists2
        result["k2"] = len(path2) - 1

    if src_ch_id == tgt_ch_id:
        result["path3"] = [src_ch_id]
        result["distances3"] = []
        result["k3"] = 0
    else:
        path3, dists3 = bfs_shortest_path_with_distance(
            graph,
            manet,
            src_ch_id,
            tgt_ch_id
        )

        if path3 is None:
            return None

        result["path3"] = path3
        result["distances3"] = dists3
        result["k3"] = len(path3) - 1

    return result


# ==================== 可靠性计算 ====================
def path_stability_probability(t, k_info):
    if k_info is None:
        return 0.0

    all_distances = (
        k_info["distances1"]
        + k_info["distances3"]
        + k_info["distances2"]
    )

    prob = 1.0

    for d in all_distances:
        prob *= link_survival_probability(t, d, v, R_ch_node)

    return prob


def transmission_success_rate(t, k_info):
    if k_info is None:
        return 0.0

    total_hops = k_info["k1"] + k_info["k3"] + k_info["k2"]

    if total_hops == 0:
        return 1.0

    num_slots = int(t // tau)
    p_data = 1.0 if num_slots >= total_hops else 0.0
    p_stable = path_stability_probability(t, k_info)

    return p_stable * p_data


def get_max_success_rate_for_pair(src_node, tgt_node, manet, graph, t_max=T_MAX, num_steps=200):
    k_info = get_k_params_multi_hop(src_node, tgt_node, manet, graph)

    if k_info is None:
        return 0.0

    time_points = np.linspace(0, t_max, num_steps)
    max_p = 0.0

    for t in time_points:
        p = transmission_success_rate(t, k_info)

        if p > max_p:
            max_p = p

        if max_p >= 1.0:
            break

    return max_p


# ==================== 单次网络实例评估 ====================
def evaluate_intra_inter_by_max_success(manet, t_max=T_MAX, num_time_steps=200):
    graph = build_communication_graph(manet)

    node_dict = {node.node_id: node for node in manet.nodes}
    node_ids = sorted(node_dict.keys())

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

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            s_id = node_ids[i]
            t_id = node_ids[j]

            s_node = node_dict[s_id]
            t_node = node_dict[t_id]

            c1 = node_to_cluster.get(s_id, None)
            c2 = node_to_cluster.get(t_id, None)

            if c1 is None or c2 is None:
                continue

            max_p = get_max_success_rate_for_pair(
                s_node,
                t_node,
                manet,
                graph,
                t_max=t_max,
                num_steps=num_time_steps
            )

            if c1 == c2:
                intra_list.append(max_p)
            else:
                inter_list.append(max_p)

    R_intra = np.mean(intra_list) if intra_list else 0.0
    R_inter = np.mean(inter_list) if inter_list else 0.0

    return R_intra, R_inter


# ==================== 统计函数 ====================
def mean_ci95(values):
    arr = np.array(values, dtype=float)

    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    se = std_val / math.sqrt(len(arr))
    ci95 = 1.96 * se

    return mean_val, ci95


# ==================== 单次 Monte Carlo ====================
def run_single_simulation(P):
    net = MANET(
        num_nodes=num_nodes,
        max_x=max_x,
        max_y=max_y,
        P=P
    )

    net.random_walk_all()
    net.select_cluster_head()
    net.make_cluster()

    R_intra, R_inter = evaluate_intra_inter_by_max_success(
        net,
        t_max=T_MAX,
        num_time_steps=200
    )

    return R_intra, R_inter


# ==================== 主程序 ====================
if __name__ == "__main__":
    all_results = []

    print(f"Starting Monte Carlo simulation: {NUM_MC} runs for each cluster-head ratio.\n")

    for P in tqdm(P_values, desc="Cluster-head ratio", colour="blue"):
        intra_reliabilities = []
        inter_reliabilities = []

        for _ in tqdm(
            range(NUM_MC),
            desc=f"P={P:.2f}",
            leave=False,
            colour="green"
        ):
            R_intra, R_inter = run_single_simulation(P)

            intra_reliabilities.append(R_intra)
            inter_reliabilities.append(R_inter)

        mean_intra, ci_intra = mean_ci95(intra_reliabilities)
        mean_inter, ci_inter = mean_ci95(inter_reliabilities)

        all_results.append({
            "Cluster_Head_Ratio": P,
            "Intra_Mean_Max_Reliability": mean_intra,
            "Intra_CI95": ci_intra,
            "Inter_Mean_Max_Reliability": mean_inter,
            "Inter_CI95": ci_inter
        })

        print(
            f"P={P:.2f} | "
            f"Intra={mean_intra:.4f} ± {ci_intra:.4f}, "
            f"Inter={mean_inter:.4f} ± {ci_inter:.4f}"
        )

    result_df = pd.DataFrame(all_results)

    output_file = "cluster_head_ratio_intra_inter_reliability_100runs.xlsx"
    result_df.to_excel(output_file, index=False)

    print(f"\nExcel saved: {output_file}")
    print("Done.")