import re
import numpy as np
from pysptools.distance import SID

def evaluate_results_metrics(true_edges, inferred_edges, metrics_list):
    """
    Evaluate the inferred edges against the true edges and compute various metrics.

    Parameters:
    - true_edges: List or Set of tuples representing the true edges (source, target).
    - inferred_edges: List or Set of tuples representing the inferred edges.
    - metrics_list: List of strings representing metrics to compute.
      Options: 'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'sensitivity', 
      'f1', 'accuracy', 'AHD', 'SHD', 'EHD', 'SID'.

    Returns:
    - A dictionary containing the computed metrics in the same order as metrics_list.
    """

    # --- 1. Helper Function for Node Normalization ---
    def map_to_x_format(node):
        """Maps 'G1', 'G2'... to 'x0', 'x1'... if necessary."""
        if isinstance(node, str) and node.startswith('G') and node[1:].isdigit():
            return f'x{int(node[1:]) - 1}'
        return node

    # --- 2. Data Preparation ---
    # Convert input edges to sets for efficient set operations
    true_edge_set = set(true_edges)
    
    # Normalize inferred edges and convert to set
    inferred_edges_normalized = [
        (map_to_x_format(u), map_to_x_format(v)) for u, v in inferred_edges
    ]
    inferred_edge_set = set(inferred_edges_normalized)

    # --- 3. Base Metrics Calculation (Counts) ---
    # Intersection = True Positives
    tp_count = len(true_edge_set & inferred_edge_set)
    # In inferred but not in true = False Positives
    fp_count = len(inferred_edge_set - true_edge_set)
    # In true but not in inferred = False Negatives
    fn_count = len(true_edge_set - inferred_edge_set)

    # Determine total nodes to calculate TN and total possible edges
    all_nodes = set()
    for edge in true_edge_set.union(inferred_edge_set):
        all_nodes.update(edge)
    
    n_nodes = len(all_nodes)
    total_possible_edges = n_nodes * (n_nodes - 1)
    
    # True Negatives
    tn_count = total_possible_edges - tp_count - fp_count - fn_count

    # --- 4. Metrics Computation ---
    results = {}

    # We iterate through the requested metrics to ensure the output order matches input
    for metric in metrics_list:
        if metric == 'TP':
            results['TP'] = tp_count
        
        elif metric == 'FP':
            results['FP'] = fp_count
        
        elif metric == 'FN':
            results['FN'] = fn_count
        
        elif metric == 'TN':
            results['TN'] = tn_count

        elif metric == 'precision':
            # Precision = TP / (TP + FP)
            denominator = tp_count + fp_count
            results['precision'] = tp_count / denominator if denominator > 0 else 0.0

        elif metric in ['recall', 'sensitivity']:
            # Recall = TP / (TP + FN)
            denominator = tp_count + fn_count
            val = tp_count / denominator if denominator > 0 else 0.0
            results[metric] = val

        elif metric in ['f1', 'f1_score']:
            # F1 = 2 * (Precision * Recall) / (Precision + Recall)
            prec_denom = tp_count + fp_count
            rec_denom = tp_count + fn_count
            
            p = tp_count / prec_denom if prec_denom > 0 else 0.0
            r = tp_count / rec_denom if rec_denom > 0 else 0.0
            
            results[metric] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        elif metric == 'accuracy':
            # Accuracy = (TP + TN) / Total
            total = tp_count + tn_count + fp_count + fn_count
            results['accuracy'] = (tp_count + tn_count) / total if total > 0 else 0.0

        elif metric == 'SHD':
            # Structural Hamming Distance = FP + FN
            results['SHD'] = fp_count + fn_count

        elif metric == 'AHD':
            # Average Hamming Distance = (FP + FN) / Total Possible
            results['AHD'] = (fp_count + fn_count) / total_possible_edges if total_possible_edges > 0 else 0.0

        elif metric == 'EHD':
            # Edge Hamming Distance (uses undirected/frozen sets to ignore direction)
            undir_true = set(frozenset(edge) for edge in true_edge_set)
            undir_inferred = set(frozenset(edge) for edge in inferred_edge_set)
            results['EHD'] = len(undir_true.symmetric_difference(undir_inferred))

        elif metric == 'SID':
            # Structural Intervention Distance
            try:
                # SID requires adjacency matrices, not edge lists.
                # Create a consistent node mapping for matrix indices
                sorted_nodes = sorted(list(all_nodes))
                node_map = {node: i for i, node in enumerate(sorted_nodes)}
                dim = len(sorted_nodes)

                # Initialize adjacency matrices
                true_matrix = np.zeros((dim, dim))
                inferred_matrix = np.zeros((dim, dim))

                # Populate matrices
                for u, v in true_edge_set:
                    if u in node_map and v in node_map:
                        true_matrix[node_map[u], node_map[v]] = 1

                for u, v in inferred_edge_set:
                    if u in node_map and v in node_map:
                        inferred_matrix[node_map[u], node_map[v]] = 1

                # Calculate SID using the matrices
                sid_value = SID(true_matrix, inferred_matrix)
                results['SID'] = sid_value
            except Exception as e:
                # Fallback or error logging if SID calculation fails
                print(f"Error calculating SID: {e}")
                results['SID'] = None

    return results