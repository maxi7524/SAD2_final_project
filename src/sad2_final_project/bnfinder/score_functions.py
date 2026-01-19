import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List

# ------------------------
# Load files
# ------------------------


def load_structure_from_sif(sif_path: str) -> Dict[str, List[str]]:
    """
    Reads a .sif file and returns parent sets.

    Returns:
        parents[node] = [parent1, parent2, ...]
    """
    parents = defaultdict(list)

    with open(sif_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                parent, _, child = parts[:3]
                parents[child].append(parent)

    return dict(parents)


def _parent_configurations(df: pd.DataFrame, parents: List[str]):
    """
    Returns all possible parent value configurations.
    """
    if not parents:
        return [()]

    parent_values = [df[p].unique() for p in parents]
    return list(itertools.product(*parent_values))


# ------------------------
# Values calculation
# ------------------------
def _count_parent_config(df: pd.DataFrame, parents: List[str], config):
    """
    Counts N_ij = number of rows matching parent configuration.
    """
    if not parents:
        return len(df)

    mask = np.ones(len(df), dtype=bool)
    for p, v in zip(parents, config):
        mask &= (df[p] == v)

    return mask.sum(), mask

def _count_node_given_parents(df, node, mask):
    """
    Returns dict: value -> N_ijk
    """
    counts = {}
    for val in df[node].unique():
        counts[val] = ((df[node] == val) & mask).sum()
    return counts


# ------------------------
# Score functions calculation
# ------------------------

def _local_scores(df: pd.DataFrame, node: str, parents: List[str]):
    """
    Computes:
        - local log-likelihood
        - number of parameters
    """
    ll = 0.0
    n_params = 0
    node_values = df[node].unique()

    for config in _parent_configurations(df, parents):
        Nij, mask = _count_parent_config(df, parents, config)
        if Nij == 0:
            continue

        counts = _count_node_given_parents(df, node, mask)

        for val, Nijk in counts.items():
            if Nijk > 0:
                ll += Nijk * np.log(Nijk / Nij)

        # parameters per parent configuration
        n_params += len(node_values) - 1

    return ll, n_params

def _compute_scores(df: pd.DataFrame, parents_map: Dict[str, List[str]]):
    """
    Computes LL, BIC, MDL for a fixed DAG.
    """
    total_ll = 0.0
    total_params = 0
    N = len(df)

    for node in df.columns:
        parents = parents_map.get(node, [])
        ll_i, p_i = _local_scores(df, node, parents)
        total_ll += ll_i
        total_params += p_i

    bic = total_ll - 0.5 * total_params * np.log(N)
    mdl = -total_ll + 0.5 * total_params * np.log(N)

    return {
        "log_likelihood": total_ll,
        "bic": bic,
        "mdl": mdl,
        "n_parameters": total_params
    }

# ------------------------
# Public functions calculation
# ------------------------

def score_dag_from_sif(dataset_df: pd.DataFrame, sif_file_path: str):
    """
    High-level function:
    - 
    - loads DAG from `sif_file_path` file path
    - computes LL, BIC, MDL

    This is the ONLY function meant to be imported.
    """
    parents_map = load_structure_from_sif(sif_file_path)
    return _compute_scores(dataset_df, parents_map)
