from typing import Tuple, List
from collections import defaultdict

import numpy as np
#from scipy import special

num_nodes = 25
epsilon = 1e-6

directed_edges = [(i-1, j-1) for i, j in [
   (1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)
]]


def build_digraph_adj_list(edges: List[Tuple]) -> np.ndarray:
    graph = defaultdict(list)
    for source, target in edges:
        graph[source].append(target)
    return graph


def normalize_incidence_matrix(im: np.ndarray, full_im: np.ndarray) -> np.ndarray:
    # NOTE:
    # 1. The paper assumes that the Incidence matrix is square,
    #    so that the normalized form A @ (D ** -1) is viable.
    #    However, if the incidence matrix is non-square, then
    #    the above normalization won't work.
    #    For now, move the term (D ** -1) to the front
    # 2. It's not too clear whether the degree matrix of the FULL incidence matrix
    #    should be calculated, or just the target/source IMs.
    #    However, target/source IMs are SINGULAR matrices since not all nodes
    #    have incoming/outgoing edges, but the full IM as described by the paper
    #    is also singular, since ±1 is used for target/source nodes.
    #    For now, we'll stick with adding target/source IMs.
    degree_mat = full_im.sum(-1) * np.eye(len(full_im))
    # Since all nodes should have at least some edge, degree matrix is invertible
    inv_degree_mat = np.linalg.inv(degree_mat)
    return (inv_degree_mat @ im) + epsilon


def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    # NOTE: For now, we won't consider all possible edges
    # max_edges = int(special.comb(num_nodes, 2))
    max_edges = len(edges)
    source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.
        target_graph[target_node, edge_id] = 1.
    full_graph = source_graph + target_graph
    source_graph = normalize_incidence_matrix(source_graph, full_graph)
    target_graph = normalize_incidence_matrix(target_graph, full_graph)
    return source_graph, target_graph


def build_digraph_adj_matrix(edges: List[Tuple]) -> np.ndarray:
    graph = np.zeros((num_nodes, num_nodes), dtype='float32')
    for edge in edges:
        graph[edge] = 1
    return graph


class Graph:
    def __init__(self):
        super().__init__()
        self.num_nodes = num_nodes
        self.edges = directed_edges
        # Incidence matrices
        self.source_M, self.target_M = \
            build_digraph_incidence_matrix(self.num_nodes, self.edges)