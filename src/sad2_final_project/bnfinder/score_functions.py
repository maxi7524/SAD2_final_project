import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.special import gammaln
import itertools
from .bnfinder_wrapper import load_structure_from_sif

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parent_config_count(df: pd.DataFrame, parents: List[str]) -> int:
    if not parents:
        return 1
    return int(np.prod([df[p].nunique() for p in parents]))


def node_cardinality(df: pd.DataFrame, node: str) -> int:
    return df[node].nunique()


# --------------------------------------------------
# Log-likelihood (MLE)
# --------------------------------------------------

def local_log_likelihood(
    df: pd.DataFrame,
    node: str,
    parents: List[str]
) -> float:
    """
    Local multinomial log-likelihood (MLE).
    """
    if parents:
        counts = (
            df.groupby(parents + [node])
              .size()
              .rename("Nijk")
              .reset_index()
        )
        Nij = (
            counts.groupby(parents)["Nijk"]
            .sum()
            .rename("Nij")
            .reset_index()
        )
        counts = counts.merge(Nij, on=parents)
    else:
        counts = (
            df.groupby([node])
              .size()
              .rename("Nijk")
              .reset_index()
        )
        Nij = counts["Nijk"].sum()
        counts["Nij"] = Nij

    return float((counts["Nijk"] * np.log2(counts["Nijk"] / counts["Nij"])).sum())


# --------------------------------------------------
# MDL (≡ BIC)
# --------------------------------------------------

def local_mdl(
    df: pd.DataFrame,
    node: str,
    parents: List[str],
    N: int
) -> float:
    """
    Local MDL contribution.
    """
    ll = local_log_likelihood(df, node, parents)

    r_i = node_cardinality(df, node)
    q_i = parent_config_count(df, parents)
    k_i = (r_i - 1) * q_i

    return -ll + 0.5 * k_i * np.log2(N)


# --------------------------------------------------
# BDe / BDeu
# --------------------------------------------------

def local_bde(
    df: pd.DataFrame,
    node: str,
    parents: List[str],
    ess: float = 1.0
) -> float:
    """
    Local BDe (BDeu) score.
    """
    r_i = node_cardinality(df, node)
    q_i = parent_config_count(df, parents)

    alpha_ijk = ess / (q_i * r_i)
    alpha_ij = ess / q_i

    if parents:
        counts = (
            df.groupby(parents + [node])
              .size()
              .rename("Nijk")
              .reset_index()
        )

        Nij = (
            counts.groupby(parents)["Nijk"]
            .sum()
            .rename("Nij")
            .reset_index()
        )

        counts = counts.merge(Nij, on=parents)

        score = 0.0
        for _, group in counts.groupby(parents):
            Nij_val = group["Nij"].iloc[0]

            score += gammaln(alpha_ij) - gammaln(alpha_ij + Nij_val)
            score += (
                gammaln(alpha_ijk + group["Nijk"])
                - gammaln(alpha_ijk)
            ).sum()

        return float(score)

    else:
        counts = (
            df.groupby([node])
              .size()
              .rename("Nijk")
              .reset_index()
        )

        Nij = counts["Nijk"].sum()

        score = gammaln(alpha_ij) - gammaln(alpha_ij + Nij)
        score += (
            gammaln(alpha_ijk + counts["Nijk"])
            - gammaln(alpha_ijk)
        ).sum()

        return float(score)


# --------------------------------------------------
# Global DAG scores
# --------------------------------------------------

def _compute_scores(
    df: pd.DataFrame,
    parents_map: Dict[str, List[str]],
    ess: float = 1.0
) -> Dict[str, float]:
    """
    Computes total LL, MDL and BDe for a fixed DAG.
    """
    N = len(df)

    total_ll = 0.0
    total_mdl = 0.0
    total_bde = 0.0

    for node in df.columns:
        parents = parents_map.get(node, [])

        total_ll -= local_log_likelihood(df, node, parents)
        total_mdl += local_mdl(df, node, parents, N)
        total_bde -= local_bde(df, node, parents, ess)

    return {
        "log_likelihood": total_ll,
        "MDL": total_mdl,
        "BDe": total_bde
    }

# ------------------------
# Public functions calculation
# ------------------------

def score_dag_from_sif(
    dataset_df: pd.DataFrame,
    sif_file_path: str,
    ess: float = 1.0
) -> Dict[str, float]:
    """
    High-level function (public API).

    - loads DAG structure from a .sif file
    - computes LL, MDL and BDe scores

    Args:
        dataset_df : pandas DataFrame with discrete variables
        sif_file_path : path to .sif file
        ess : equivalent sample size for BDe (default = 1.0)

    Returns:
        dict with keys:
            - log_likelihood
            - MDL
            - BDe
    """
    parents_map = load_structure_from_sif(sif_file_path)
    return _compute_scores(dataset_df, parents_map, ess)
