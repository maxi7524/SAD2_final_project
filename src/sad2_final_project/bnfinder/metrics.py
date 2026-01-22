import re

def evaluate_results_metrics(true_edges, inferred_edges, metrics_list):
    """
    Evaluate the inferred edges against the true edges and compute various metrics.

    Parameters:
    - true_edges: Set of tuples representing the true edges in the format (source, target).
    - inferred_edges: Set of tuples representing the inferred edges in the format (source, target).
    - metrics_list: List of metrics to compute. Options include 'TP', 'FP', 'FN', 'precision', 'recall', 'sensitivity', 'AHD', 'SHD', 'EHD', 'SID'.

    Returns:
    - A dictionary containing the computed metrics in the same order as metrics_list.
    """
    # Convert edges to sets for easier computation
    true_edge_set = set(true_edges)

    # Parse inferred edges to match the format of true edges
    inferred_edge_set = set()
    for edge in inferred_edges:
        # Extract nodes from the format 'G<int a> -> G<int b>' and map 'Gn' to 'x{n-1}'
        ###### zmiana bo split sie wywalał
        # print("start split")
        # parts = edge.split(" -> ")
        # print("stop split")
        ####### czy to zawsze będzie 3?
        if len(edge) == 3:
            source = edge[0].strip()
            target = edge[1].strip()
            # Replace 'Gn' with 'x{n-1}'
            source = re.sub(r'G(\d+)', lambda m: f"x{int(m.group(1)) - 1}", source)
            target = re.sub(r'G(\d+)', lambda m: f"x{int(m.group(1)) - 1}", target)
            inferred_edge_set.add((source, target))
    inferred_edge_set = set([(f'x{el_1[1]-1}', f'x{el_2[1]-1}') for el_1, el_2 in inferred_edges])

    def calculate_tp():
        return len(true_edge_set & inferred_edge_set)

    def calculate_fp():
        return len(inferred_edge_set - true_edge_set)

    def calculate_fn():
        return len(true_edge_set - inferred_edge_set)

    def calculate_precision(tp, fp):
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def calculate_recall(tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def calculate_sensitivity(recall):
        return recall  # Sensitivity is equivalent to recall

    ####### true_edges + inferred_edges tu sie wywalało że jeden to set drugi list
    # def calculate_ahd(fp, fn):
    #     all_nodes = set()
    #     for edge in true_edges + inferred_edges:
    #         all_nodes.update(edge)
    #     n = len(all_nodes)
    #     total_possible_edges = n * (n - 1)
    #     return (fp + fn) / total_possible_edges if total_possible_edges > 0 else 0.0
    
    def calculate_ahd(fp, fn):
        all_nodes = set()
        for edge in true_edge_set.union(inferred_edge_set):
            all_nodes.update(edge)
        n = len(all_nodes)
        total_possible_edges = n * (n - 1)
        return (fp + fn) / total_possible_edges if total_possible_edges > 0 else 0.0

    def calculate_shd(fp, fn):
        return fp + fn

    def calculate_ehd():
        undir_true_edges = set(frozenset(edge) for edge in true_edges)
        undir_inferred_edges = set(frozenset(edge) for edge in inferred_edges)
        return len(undir_true_edges.symmetric_difference(undir_inferred_edges))

    def calculate_sid():
        directed_differences = 0
        for edge in inferred_edges:
            if edge[::-1] in true_edges and edge not in true_edges:
                directed_differences += 1
        return directed_differences

    # Compute metrics conditionally
    metric_values = {}

    # TODO - PACKAGE: write some function that wraps it up 
    if 'TP' in metrics_list:
        metric_values['TP'] = calculate_tp()
    if 'FP' in metrics_list:
        metric_values['FP'] = calculate_fp()
    if 'FN' in metrics_list:
        metric_values['FN'] = calculate_fn()
    if 'precision' in metrics_list:
        TP = metric_values.get('TP', calculate_tp())
        FP = metric_values.get('FP', calculate_fp())
        metric_values['precision'] = calculate_precision(TP, FP)
    if 'recall' in metrics_list:
        TP = metric_values.get('TP', calculate_tp())
        FN = metric_values.get('FN', calculate_fn())
        metric_values['recall'] = calculate_recall(TP, FN)
    if 'sensitivity' in metrics_list:
        recall = metric_values.get('recall', calculate_recall(
            metric_values.get('TP', calculate_tp()),
            metric_values.get('FN', calculate_fn())
        ))
        metric_values['sensitivity'] = calculate_sensitivity(recall)
    if 'AHD' in metrics_list:
        FP = metric_values.get('FP', calculate_fp())
        FN = metric_values.get('FN', calculate_fn())
        metric_values['AHD'] = calculate_ahd(FP, FN)
    if 'SHD' in metrics_list:
        FP = metric_values.get('FP', calculate_fp())
        FN = metric_values.get('FN', calculate_fn())
        metric_values['SHD'] = calculate_shd(FP, FN)
    if 'EHD' in metrics_list:
        metric_values['EHD'] = calculate_ehd()
    if 'SID' in metrics_list:
        metric_values['SID'] = calculate_sid()

    # Prepare results dictionary in the same order as metrics_list
    results = {metric: metric_values[metric] for metric in metrics_list if metric in metric_values}

    return results