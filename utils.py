"""
Utility functions for MrNEAD demo
"""

import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os
import community.community_louvain as community_louvain
from collections import defaultdict

def load_network(network_name):
    """
    Load network by name
    
    Parameters:
    -----------
    network_name : str, network name (e.g., "football")
    
    Returns:
    --------
    G : networkx.Graph
    A : numpy.ndarray, adjacency matrix
    """
    data_path = f"data/{network_name}.txt"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Network file not found: {data_path}\n"
            f"Please ensure the network file exists in the data/ directory."
        )
    
    G = create_graph_from_file(data_path)
    print(f"   Loaded {network_name} from: {data_path}")

    # Convert to adjacency matrix
    A = nx.adjacency_matrix(G).toarray().astype(float)
    
    return G, A

def create_graph_from_file(filename):
    """Create NetworkX graph from edge list file"""
    G = nx.Graph()
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            nodes = line.strip().split()
            if len(nodes) >= 2:
                edge = (int(nodes[0]), int(nodes[1]))
                G.add_edge(*edge)
    
    return G


def louvain_communities(G):
    """Convert Louvain dict partition -> list of communities"""
    partition = community_louvain.best_partition(G)  # dict: node -> community id

    comm_dict = defaultdict(list)
    for node, cid in partition.items():
        comm_dict[cid].append(node)

    return list(comm_dict.values())


def evaluate_attack(A_original, A_attacked):
    """
    Evaluate the effectiveness of the attack using Louvain community detection
    """
    # Convert to NetworkX graphs
    G_original = nx.from_numpy_array(A_original)
    G_attacked = nx.from_numpy_array(A_attacked)

    # Louvain community detection
    original_communities = louvain_communities(G_original)
    attacked_communities = louvain_communities(G_attacked)

    # Compute modularity
    original_modularity = community_louvain.modularity(
        community_louvain.best_partition(G_original), G_original
    )
    attacked_modularity = community_louvain.modularity(
        community_louvain.best_partition(G_attacked), G_attacked
    )

    # Convert communities to labels for NMI/ARI calculation
    n_nodes = A_original.shape[0]
    original_labels = community_to_labels(original_communities, n_nodes)
    attacked_labels = community_to_labels(attacked_communities, n_nodes)

    # Compute NMI and ARI
    nmi_score = normalized_mutual_info_score(original_labels, attacked_labels)
    ari_score = adjusted_rand_score(original_labels, attacked_labels)

    return {
        'original_modularity': original_modularity,
        'attacked_modularity': attacked_modularity,
        'modularity_reduction': original_modularity - attacked_modularity,
        'nmi_louvain': nmi_score,
        'ari_louvain': ari_score,
        'original_communities': len(original_communities),
        'attacked_communities': len(attacked_communities)
    }


def community_to_labels(communities, n_nodes):
    """Convert community list to node labels array"""
    labels = np.zeros(n_nodes, dtype=int)
    
    for i, community in enumerate(communities):
        for node in community:
            labels[node] = i
    
    return labels
