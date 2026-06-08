import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==================== 图形字体设置 ====================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# ==================== 参数设置 ====================
MAX_X = 100
MAX_Y = 100
NUM_NODES = 50
P = 0.1
NUM_CH = int(NUM_NODES * P)

V = 2.0
R_MEMBER = 30.0
R_CH = 50.0

E_INIT_MEAN = 30.0
E_INIT_STD = 5.0

E_CH_AVG = 1.0
E_NODE_COMM = 0.5

TIME_POINTS = [5, 10, 15, 20]

COLORS = [
    "#1049D8",   # Cluster A 蓝色
    "#F0321A",   # Cluster B 红色
    "#12B24B",   # Cluster C 绿色
    "#7A2CA7",   # Cluster D 紫色
    "#F5B800"    # Cluster E 金黄色
]

CLUSTER_NAMES = ["A", "B", "C", "D", "E"]


# ==================== 节点类 ====================
class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.x = random.uniform(0, MAX_X)
        self.y = random.uniform(0, MAX_Y)
        self.direction = random.uniform(0, 2 * math.pi)

        self.initial_energy = max(1.0, np.random.normal(E_INIT_MEAN, E_INIT_STD))
        self.energy = self.initial_energy

        self.is_cluster_head = False
        self.cluster_head = None
        self.last_ch_round = -999

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def move(self):
        self.x += V * math.cos(self.direction)
        self.y += V * math.sin(self.direction)

        if self.x < 0 or self.x > MAX_X:
            self.direction = math.pi - self.direction
            self.x = max(0, min(MAX_X, self.x))

        if self.y < 0 or self.y > MAX_Y:
            self.direction = -self.direction
            self.y = max(0, min(MAX_Y, self.y))

    def reset_cluster_state(self):
        self.is_cluster_head = False
        self.cluster_head = None

    def ch_selection_score(self, r):
        """
        式(5.1):
        T(n)= P/[1-P(r mod floor(1/P))] * E_res/E_i

        当 P=0.1 时：
        T(n)=1/[10-(r mod 10)] * E_res/E_i
        """
        cycle_length = int(1 / P)
        r_mod = r % cycle_length

        base_prob = P / (1 - P * r_mod)
        energy_ratio = self.energy / self.initial_energy

        return base_prob * energy_ratio


# ==================== 网络类 ====================
class MANET:
    def __init__(self):
        self.nodes = [Node(i) for i in range(NUM_NODES)]
        self.cluster_heads = []
        self.round = 0

    def select_cluster_heads(self):
        for node in self.nodes:
            node.reset_cluster_state()

        candidates = sorted(
            self.nodes,
            key=lambda n: n.ch_selection_score(self.round),
            reverse=True
        )

        self.cluster_heads = candidates[:NUM_CH]

        for ch in self.cluster_heads:
            ch.is_cluster_head = True
            ch.last_ch_round = self.round

    def distance_factor(self, node, ch):
        """
        式(5.3):
        F(d)=(R-d)/R, 0<d<R
        F(d)=0, d>=R
        """
        d = node.distance(ch)

        if 0 < d < R_CH:
            return (R_CH - d) / R_CH
        else:
            return 0.0

    def current_member_count(self, ch_index):
        return sum(
            1 for node in self.nodes
            if node.cluster_head == ch_index
        )

    def leadership_capacity(self, ch_index):

        ch = self.cluster_heads[ch_index]

        k_i = self.current_member_count(ch_index)

        return max(
            0.0,
            ch.initial_energy - k_i
        )

    def make_clusters(self):
        """
        式(5.2):
        P_j->i = k(E_i)F(d_j->i) / sum[k(E_l)F(d_j->l)]
        """
        for node in self.nodes:
            if node.is_cluster_head:
                node.cluster_head = None
                continue

            scores = []

            for i, ch in enumerate(self.cluster_heads):
                k_e = self.leadership_capacity(i)
                f_d = self.distance_factor(node, ch)
                scores.append(k_e * f_d)

            total_score = sum(scores)

            if total_score <= 0:
                node.cluster_head = None
            else:
                probs = np.array(scores) / total_score
                node.cluster_head = int(np.random.choice(range(NUM_CH), p=probs))
                #node.cluster_head = int(np.argmax(probs))

    def update_energy(self):
        """
        能量消耗：
        簇头平均能耗：1 J/个
        普通节点通信能耗：0.5 J/次
        """
        for ch in self.cluster_heads:
            ch.energy = max(0.0, ch.energy - E_CH_AVG)

        for node in self.nodes:
            if not node.is_cluster_head and node.cluster_head is not None:
                node.energy = max(0.0, node.energy - E_NODE_COMM)
                ch = self.cluster_heads[node.cluster_head]
                ch.energy = max(0.0, ch.energy - E_NODE_COMM)

    def move_all(self):
        for node in self.nodes:
            node.move()

    def one_round(self):
        self.round += 1
        self.move_all()
        self.select_cluster_heads()
        self.make_clusters()
        self.update_energy()

    def cluster_statistics(self):
        stats = []

        for i in range(NUM_CH):
            count = 1
            count += sum(
                1 for node in self.nodes
                if node.cluster_head == i
            )
            stats.append(count)

        detached = sum(
            1 for node in self.nodes
            if not node.is_cluster_head and node.cluster_head is None
        )

        return stats, detached


# ==================== 绘图 ====================
def plot_snapshots(snapshots):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, snapshot in zip(axes, snapshots):
        t, nodes, ch_ids = snapshot

        id_to_node = {node.node_id: node for node in nodes}
        cluster_heads = [id_to_node[ch_id] for ch_id in ch_ids]

        # 绘制簇头通信范围
        for ch in cluster_heads:
            circle = plt.Circle(
                (ch.x, ch.y),
                R_CH,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                color="black",
                alpha=0.8
            )
            ax.add_patch(circle)

        # 绘制节点
        for node in nodes:
            if node.is_cluster_head:
                ch_index = ch_ids.index(node.node_id)
                ax.scatter(
                    node.x,
                    node.y,
                    marker="*",
                    s=220,
                    color=COLORS[ch_index],
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4
                )
                ax.text(
                    node.x + 1,
                    node.y + 1,
                    f"CH.{CLUSTER_NAMES[ch_index]}:{node.node_id}",
                    fontsize=9
                )

            elif node.cluster_head is None:
                ax.scatter(
                    node.x,
                    node.y,
                    marker="o",
                    s=35,
                    color="black",
                    zorder=3
                )

            else:
                ax.scatter(
                    node.x,
                    node.y,
                    marker="o",
                    s=35,
                    color=COLORS[node.cluster_head],
                    zorder=3
                )

        ax.set_xlim(0, MAX_X)
        ax.set_ylim(0, MAX_Y)
        ax.set_aspect("equal")
        ax.set_title(f"t={t}s", fontsize=14)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 20, 40, 60, 80, 100])

    # 图例
    legend_elements = [
        Line2D(
            [0], [0],
            marker="*",
            linestyle="--",
            color="black",
            markersize=14,
            label="CH and its leadership"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[0],
            markersize=14,
            label="Cluster A"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[1],
            markersize=14,
            label="Cluster B"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[2],
            markersize=14,
            label="Cluster C"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[3],
            markersize=14,
            label="Cluster D"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[4],
            markersize=14,
            label="Cluster E"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=14,
            label="Isolated node"
        )
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    # 保存为SVG矢量图
    plt.savefig(
        "Fig6_CMANET.svg",
        format="svg",
        bbox_inches="tight"
    )

    # 保存为高分辨率PNG
    plt.savefig(
        "Fig6_CMANET.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()


def save_snapshot(net, t):
    nodes_copy = copy.deepcopy(net.nodes)
    ch_ids = [ch.node_id for ch in net.cluster_heads]
    return t, nodes_copy, ch_ids


# ==================== 主程序 ====================
if __name__ == "__main__":
    #random.seed(42)
    #np.random.seed(42)

    net = MANET()
    snapshots = []

    print("表 5 移动自组织网络的簇团结构参数")
    print("时间\tCluster A\tCluster B\tCluster C\tCluster D\tCluster E\tIsolated node")

    for t in range(1, max(TIME_POINTS) + 1):
        net.one_round()

        if t in TIME_POINTS:
            stats, detached = net.cluster_statistics()

            print(
                f"t={t}s\t"
                f"{stats[0]}\t\t"
                f"{stats[1]}\t\t"
                f"{stats[2]}\t\t"
                f"{stats[3]}\t\t"
                f"{stats[4]}\t\t"
                f"{detached}"
            )

            snapshots.append(save_snapshot(net, t))

    plot_snapshots(snapshots)