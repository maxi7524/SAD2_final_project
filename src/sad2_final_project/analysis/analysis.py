# data manipulation
import pandas as pd
import numpy as np
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
# data analysis
from scipy.stats import wilcoxon, spearmanr
# helpers
from pathlib import Path
from typing import List

from sad2_final_project.analysis import experiment

# ---------------------------
# Global style
# ---------------------------
plt.rcParams['font.family'] = 'Liberation Serif'
plt.rcParams['font.size'] = 16
plt.rcParams['text.color'] = '#2C3E50'

# example category colors
category_colors = [
    '#C0392B', 
    '#2980B9', 
    '#27AE60', 
    '#F4D03F', 
    '#8E44AD'
    ]

# TODO JOANNA: sprawdź czy lepiej ci nie pasują - widać wtedy skalowaność node'ów
category_colors = [
    "#e0d9e0",  # najmniejsze num_nodes
    "#d3a6b3",
    "#9b5f86",
    "#5f3b66",
    "#2b1e3a"   # największe num_nodes
]



# --------------------------------------------------
# MAx
# --------------------------------------------------


# --------------------
# Loading data
# --------------------



# CURRENT DATA

def experiment_data_loader(experiment_path: Path | str) -> pd.DataFrame:
    '''
    Loads data in obsolete format into one dataframe
    '''
    ## Obtain paths for csv
    experiment_path = Path(experiment_path)
    metadata_path = experiment_path / 'results/metadata.csv'
    results_path = experiment_path / 'results/results.csv'

    ## metadata
    ### load metadata into df
    metadata_df = pd.read_csv(
        metadata_path, 
        # set incorrectly read datatype (ref)
        dtype={
            "condition_id_name": str,
        })
    ## results
    ### load results 
    results_df = pd.read_csv(
        results_path,
        )

    ## merge 
    ### remove duplicates
    results_df = results_df.drop(columns=['score_function'])
    ### merge be outer join
    df = results_df.merge(metadata_df, how='outer', on='condition_id_num')
    return df

# OBSOLETE DATA
def loader_obsolete_data(metadata_path: Path | str, results_path: Path | str) -> pd.DataFrame:
    '''
    Loads data in obsolete format into one dataframe
    '''
    ## metadata
    ### load metadata into df
    metadata_df = pd.read_csv(
        metadata_path, 
        # del obsolete columns
        usecols=lambda x: x not in ['Unnamed: 0', 'attractor_ratio', 'success'],
        # set incorrectly read datatype (refe)
        dtype={
            "condition_id": str,
        })
    
    print(metadata_df.columns)
    ### rename columns in meta data to obtain 
    metadata_df['condition_id_name'] = metadata_df['condition_id']
    metadata_df['condition_id_num'] = metadata_df['condition_id'].astype(int)
    metadata_df = metadata_df.drop(columns=['condition_id'])

    ## results
    ### load results 
    results_df = pd.read_csv(
        results_path,
        usecols = lambda x: x not in ["BIC"],
        )
    ### rename columns to match with meta data
    results_df = results_df.rename(
        columns={
            "dataset": "condition_id_num",
            "score": "score_function"
        }
    )
    ### add missing values

    ## merge 
    ### remove duplicates
    results_df = results_df.drop(columns=['score_function'])
    ### merge be outer join
    df = results_df.merge(metadata_df, how='outer', on='condition_id_num')
    return df

# --------------------------------------------------
# Updating data
# --------------------------------------------------

# TODO DONE - TO HELPERS
def add_missing_metrics_from_experiment(
    df: pd.DataFrame,
    experiment_path: str | Path,
    metrics_list: List[str],
    after_column: str,
) -> pd.DataFrame:
    """
    Takes a DataFrame with results, calculates missing metrics by loading ground truth
    and inferred edges from experiment folder structure, and returns DataFrame with all metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing results with 'dataset' column (condition_id).
        Should already have columns for metrics_list (but may have NaN values).
    
    experiment_path : str | Path
        Path to the experiment folder. Function expects this structure:
        - experiment_path/bn_ground_truth/  (contains .csv files with ground truth edges)
        - experiment_path/results/          (contains .sif files with inferred edges)
    
    metrics_list : List[str]
        List of metrics in desired order. Options: 'TP', 'FP', 'FN', 'precision', 
        'recall', 'sensitivity', 'AHD', 'SHD', 'EHD', 'SID'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing metric values filled in existing columns.
    """
    
    # Lazy imports to avoid circular imports
    from sad2_final_project.analysis.metrics import evaluate_results_metrics
    from sad2_final_project.bnfinder.bnfinder import load_ground_truth
    from sad2_final_project.bnfinder.bnfinder_wrapper import parse_sif_results
    
    # Convert to Path object
    experiment_path = Path(experiment_path)
    
    # Define subdirectories
    ground_truth_dir = experiment_path / 'bn_ground_truth'
    results_dir = experiment_path / 'results'
    
    # Validate directories exist
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # create new df for results
    ## create copy of dataframe
    df_result = df.copy()
    ## change order of columns
    ### remove all metrics
    common_metrics = list(set(df.columns).intersection(set(metrics_list)))
    df_result = df_result.drop(columns=common_metrics)
    ### add metrics columns as dtype float
    for col in metrics_list:
        df_result[col] = pd.Series(dtype='float')
    ### move them to proper index 
    cols = list(df_result.columns)
    idx = cols.index(after_column)

    new_cols = cols[:idx+1] \
           + metrics_list \
           + [c for c in cols if c not in metrics_list and c not in cols[:idx+1]]

    df_result = df_result[new_cols]

    
    # Process each row - calculate missing metrics
    _counter = 0
    for idx, row in df_result.iterrows():
        dataset_id = row.get('condition_id_name')
        
        # Load ground truth
        ## check file existance 
        gt_path = ground_truth_dir / f'{dataset_id}.csv'
        if not gt_path.exists():
            print(gt_path)
            print(f"  [Warning] Ground truth not found for {dataset_id}")
            continue
        ## load true edges
        true_edges = load_ground_truth(str(gt_path))
        ## check if it is not none 
        if true_edges is None:
            print(f"  [Warning] Could not load ground truth for {dataset_id}")
            continue
        
        # Lad infer edges
        ## scan through folder with result and check if it is MDL or BDE
        inferred_edges = None
        for score_suffix in ['_MDL', '_BDE']:
            results_path = results_dir / f'{dataset_id}{score_suffix}.sif'
            if results_path.exists():
                inferred_edges = parse_sif_results(str(results_path))
                break
        ## check if returned value exists
        if inferred_edges is None:
            print(f"  [Warning] Inferred edges not found for {dataset_id}")
            continue
        
        # Calculate metrics - pass metrics_list so they're computed in correct order
        try:
            metrics = evaluate_results_metrics(true_edges, inferred_edges, metrics_list=metrics_list)
            # Insert calculated metrics into existing columns
            for metric_name, metric_value in metrics.items():
                if metric_name in df_result.columns:
                    df_result.at[idx, metric_name] = metric_value
                
        except Exception as e:
            print(f"  [Error] Failed to calculate metrics for {dataset_id}: {e}")
            continue
            
        _counter += 1
        if _counter % 1000 == 0:
            total = df_result.shape[0]
            print(f"[Progress] {_counter}/{total} conditions completed ({100*_counter/total:.1f}%)")
        
    return df_result


# TODO LIBRARY: add score functions to existing data


# --------------------
# Analysis 2
# --------------------

# TODO - to jest część analityczna (do przeniesienia)
# ----------------------------------------
# ACF AND ESS analysis
# ----------------------------------------

def acf(x, max_lag):
    """
    Compute autocorrelation function up to max_lag.
    
    Parameters
    ----------
    x : array-like, shape (T,)
        Time series.
    max_lag : int
        Maximum lag k.
    
    Returns
    -------
    acf_values : np.ndarray, shape (max_lag + 1,)
        acf_values[k] = autocorrelation at lag k.
        acf_values[0] = 1.
    """
    x = np.asarray(x)
    T = len(x)
    
    x_mean = np.mean(x)
    x_centered = x - x_mean
    
    denom = np.sum(x_centered ** 2)
    if denom == 0:
        raise ValueError("Zero variance time series.")
    
    acf_values = np.empty(max_lag + 1)
    acf_values[0] = 1.0
    
    for k in range(1, max_lag + 1):
        num = np.sum(
            x_centered[k:] * x_centered[:-k]
        )
        acf_values[k] = num / denom
    
    return acf_values

def effective_sample_size(x, max_lag=None):
    """
    Estimate effective sample size (ESS) of a time series.
    
    Parameters
    ----------
    x : array-like, shape (T,)
        Time series.
    max_lag : int or None
        Maximum lag to consider. If None, defaults to T//2.
    
    Returns
    -------
    ess : float
        Effective sample size.
    """
    x = np.asarray(x)
    T = len(x)
    
    if max_lag is None:
        max_lag = T // 2
    
    rho = acf(x, max_lag)
    
    # sum only positive autocorrelations
    positive_rho = rho[1:][rho[1:] > 0]
    
    ess = T / (1 + 2 * np.sum(positive_rho))
    
    return ess

def acf_and_ess_for_series(x, max_lag=None):
    """
    Compute ACF and ESS for a single time series.
    
    Returns
    -------
    result : dict
        {
            "ess": float,
            "acf": np.ndarray
        }
    """
    acf_values = acf(x, max_lag=max_lag)
    ess_value = effective_sample_size(x, max_lag=max_lag)
    
    return {
        "ess": ess_value,
        "acf": acf_values
    }

def dataset_level_acf_and_ess(df, max_lag=None):
    """
    Compute dataset-level mean lag-1 ACF and mean ESS.
    """
    rho1_values = []
    ess_values = []
    
    for col in df.columns:
        x = df[col].values
        
        rho = acf(x, max_lag=1)
        rho1_values.append(rho[1])
        
        ess = effective_sample_size(x, max_lag=max_lag)
        ess_values.append(ess)
    
    return {
        "mean_lag1_acf": float(np.mean(rho1_values)),
        "mean_ess": float(np.mean(ess_values))
    }

def analyze_datasets_from_index(
    meta_df,
    index_column,
    experiment_path,
    max_lag=None
):
    records = []
    
    _counter = 0
    for idx, row in meta_df.iterrows():
        dataset_name = row[index_column]
        
        dataset_path = (
            Path(experiment_path) / "datasets" / f"{dataset_name}.csv"
        )
        df = pd.read_csv(dataset_path)
        
        stats = dataset_level_acf_and_ess(df, max_lag=max_lag)
        
        records.append({
            index_column: dataset_name,
            "mean_lag1_acf": stats["mean_lag1_acf"],
            "mean_ess": stats["mean_ess"]
        })

        _counter += 1
        if _counter % 1000:
          total = meta_df.shape[0]
          print(f"[Progress] {_counter}/{total} conditions completed ({100*_counter/total:.1f}%)")

    
    return meta_df.merge(pd.DataFrame(records), how='outer', on=index_column)

# ----------------------------------------
# Statistical functions - basic functions
# ----------------------------------------


def signif_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def paired_wilcoxon(
    df,
    metric,
    sf_from,
    sf_to,
    id_cols=("num_nodes", "update_mode", "score_function", "trajectory_length"),
    sf_col="sampling_frequency",
    alternative="greater"
):
    """
    Parowany test Wilcoxona: sf_to vs sf_from
    H1 (domyślnie): sf_to < sf_from  (im mniejsza metryka, tym lepiej)
    """

    df1 = df[df[sf_col] == sf_from]
    df2 = df[df[sf_col] == sf_to]

    merged = df1.merge(
        df2,
        on=list(id_cols),
        suffixes=(f"_{sf_from}", f"_{sf_to}")
    )

    if len(merged) < 10:
        return None

    x = merged[f"{metric}_{sf_from}"]
    y = merged[f"{metric}_{sf_to}"]

    stat, p = wilcoxon(
        x - y,
        alternative=alternative
    )

    return {
        "metric": metric,
        "sf_from": sf_from,
        "sf_to": sf_to,
        "transition": f"{sf_from}→{sf_to}",
        "n_pairs": len(merged),
        "wilcoxon_stat": stat,
        "p_value": p,
        "median_diff": np.median(y - x)
    }

def spearman_analysis(df, metric, x_col="mean_ess"):
    """
    Spearman correlation between ESS and metric.
    """
    rho, p = spearmanr(df[x_col], df[metric])
    return rho, p

# ----------------------------------------
# Statistical functions - getting grouped table
# ----------------------------------------

def compute_wilcoxon_table(
    df,
    metrics,
    transitions,
    group_cols=("update_mode", "score_function"),
    sf_col="sampling_frequency"
):
    """
    Liczy wszystkie testy Wilcoxona i zwraca DataFrame.
    """

    results = []

    for group_vals, df_sub in df.groupby(list(group_cols)):
        group_dict = dict(zip(group_cols, group_vals))

        # ###### 
        # df_sub[df_sub["k_value"].isin([20, 40])][["condition_id_num", "rep_id", "k_value", "AHD"]].head(20)

        for metric in metrics:
            for sf_from, sf_to in transitions:

                res = paired_wilcoxon(
                    df_sub,
                    metric=metric,
                    sf_from=sf_from,
                    sf_to=sf_to,
                    sf_col=sf_col
                )

                if res is not None:
                    res.update(group_dict)
                    results.append(res)

    return pd.DataFrame(results)

def compute_spearman_table(
    df,
    metrics,
    group_cols=("update_mode", "score_function", "num_nodes"),
    ess_col="mean_ess"
):
    """
    Liczy korelacje Spearmana ESS vs metryka.
    """

    rows = []

    for group_vals, df_sub in df.groupby(list(group_cols)):
        group_dict = dict(zip(group_cols, group_vals))

        for metric in metrics:
            rho, p = spearman_analysis(df_sub, metric, x_col=ess_col)
            rows.append({
                **group_dict,
                "metric": metric,
                "spearman_rho": rho,
                "p_value": p
            })

    return pd.DataFrame(rows)


# TODO - to jest część od wizualizacji
# ----------------------------------------
# Plots 
# ----------------------------------------

def plot_grouped_boxplots(
    df,
    *,
    group_col,              # osobne figury, np. (update_mode)
    x_col,                  # wartość na osi X, np. "sampling_frequency"
    y_cols,                 # wartośc na si Y, np. ["AHD", "SID"] albo ["mean_lag1_acf", "mean_ess"]
    hue_col,                # jakie kategoryzuje, np. "num_nodes"
    hue_palette=category_colors,            # odcień kategorii 
    facet_col=category_colors,         # zmienne panelowe, np. "score_function"
    facet_levels=None,      # kolejność paneli np. ["MDL", "BDE"]
    main_title="",          # tytuł całej figury
    group_title_fmt="",     # np. "Update mode = {}"
    share_y_per_metric=True,
    figsize=(15, 5),
    padding_frac=0.05
):
    """
    Uniwersalna funkcja do rysowania zestawów boxplotów z pełną parametryzacją.
    """

    plt.rcParams["font.family"] = "Liberation Serif"

    hue_levels = sorted(df[hue_col].unique())

    # ============================================================
    # GLOBALNE ZAKRESY Y (dla każdej metryki osobno)
    # ============================================================
    global_y_ranges = {}
    for y in y_cols:
        ymin, ymax = df[y].min(), df[y].max()
        pad = padding_frac * (ymax - ymin)
        global_y_ranges[y] = (ymin - pad, ymax + pad)

    # ============================================================
    # GŁÓWNA PĘTLA PO GRUPACH (np. update_mode)
    # ============================================================
    if group_col is None:
        group_values = [None]
    else:
        group_values = df[group_col].unique()

    for group_val in group_values:

        if group_col is None:
            df_group = df
        else:
            df_group = df[df[group_col] == group_val]


        for y in y_cols:

            # ===== liczba subplotów =====
            n_facets = len(facet_levels) if facet_col else 1

            fig, axes = plt.subplots(
                1,
                n_facets,
                figsize=figsize,
                sharey=share_y_per_metric
            )

            if n_facets == 1:
                axes = [axes]

            # ====================================================
            # RYSOWANIE POSZCZEGÓLNYCH PANELI
            # ====================================================
            for ax, facet_val in zip(axes, facet_levels or [None]):

                if facet_col:
                    df_plot = df_group[df_group[facet_col] == facet_val]
                    title = f"{y} vs {x_col}, {facet_col} = {facet_val}"
                else:
                    df_plot = df_group
                    title = f"{y} vs {x_col}"

                plot_boxplot(
                    df=df_plot,
                    x=x_col,
                    y=y,
                    hue=hue_col,
                    palette=hue_palette,
                    title=title,
                    ax=ax,
                    show_legend=False
                )

                ax.set_ylim(*global_y_ranges[y])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y)

            # ====================================================
            # LEGENDA (jedna na całą figurę)
            # ====================================================
            handles = [
                mpatches.Patch(color=hue_palette[i], label=str(hue_levels[i]))
                for i in range(len(hue_levels))
            ]

            fig.legend(
                handles=handles,
                title=hue_col,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(hue_levels),
                frameon=False
            )

            # ====================================================
            # TYTUŁY
            # ====================================================
            if group_col is None:
                full_title = main_title
            else:
                full_title = f"{main_title}\n{group_title_fmt.format(group_val)}"

            
            fig.suptitle(full_title, fontsize=18, y=1.12)


            fig.subplots_adjust(
                top=0.78,
                wspace=0.25
            )

            plt.show()

# ----------------------------------------
# Plots - statistical
# ----------------------------------------

## basic function
def plot_stat_heatmap(
    df,
    *,
    value_col,
    p_col,
    index_col,
    column_col,
    index_order=None,
    vmin=None,
    vmax=None,
    center=None,
    cmap="coolwarm",
    value_fmt="{:.2f}",
    signif_func=None,
    figsize=(6, 5),
    ax=None,
    cbar=True,
    cbar_label=None,
    title=None,
    xlabel=None,
    ylabel=None
):
    """
    Uniwersalna funkcja do rysowania heatmap statystycznych
    (Wilcoxon, Spearman, cokolwiek z value + p-value).
    """

    pivot_val = df.pivot_table(
        index=index_col,
        columns=column_col,
        values=value_col
    )

    pivot_p = df.pivot_table(
        index=index_col,
        columns=column_col,
        values=p_col
    )

    if index_order is not None:
        pivot_val = pivot_val.reindex(index_order)
        pivot_p = pivot_p.reindex(index_order)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        pivot_val,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=0.5,
        linecolor="white",
        annot=False,
        cbar=cbar,
        cbar_kws={"label": cbar_label} if cbar_label else None
    )

    # ===== ADNOTACJE =====
    for y, idx in enumerate(pivot_val.index):
        for x, col in enumerate(pivot_val.columns):
            val = pivot_val.loc[idx, col]
            p = pivot_p.loc[idx, col]

            if pd.notna(val):
                stars = signif_func(p) if signif_func else ""
                label = value_fmt.format(val) + stars

                ax.text(
                    x + 0.5,
                    y + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black"
                )

    if title:
        ax.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.tick_params(axis="both", which="both", length=0)

    return fig, ax

## wilcoxon aggregation
def plot_wilcoxon_heatmap(
    df,
    *,
    metric,
    update_mode,
    transitions_order,
    ax=None,
    cbar=True,
    vmin=None,
    vmax=None,
):
    df_sub = df[
        (df["metric"] == metric) &
        (df["update_mode"] == update_mode)
    ]

    return plot_stat_heatmap(
        df_sub,
        value_col="median_diff",
        p_col="p_value",
        index_col="transition",
        column_col="score_function",
        index_order=transitions_order,
        vmin=vmin,
        vmax=vmax,
        center=0,
        cmap="coolwarm",
        value_fmt="{:.2f}",
        signif_func=signif_stars,
        cbar=cbar,
        cbar_label="Median difference (lower = improvement)",
        title=metric,
        xlabel="Score function",
        ylabel="Sampling frequency transition",
        ax=ax
    )
## spearman aggregation
def plot_spearman_heatmap(
    df,
    *,
    metric,
    update_mode,
    num_nodes_order,
    ax=None,
    cbar=True
):
    df_sub = df[
        (df["metric"] == metric) &
        (df["update_mode"] == update_mode)
    ]

    return plot_stat_heatmap(
        df_sub,
        value_col="spearman_rho",
        p_col="p_value",
        index_col="num_nodes",
        column_col="score_function",
        index_order=num_nodes_order,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        value_fmt="{:.2f}",
        signif_func=signif_stars,
        cbar=cbar,
        cbar_label="Spearman ρ",
        title=metric,
        xlabel="Score function",
        ylabel="Number of nodes",
        ax=ax
    )


# --------------------------------------------------
# Joanna
# --------------------------------------------------

def plot_scatter(df, x, y, title):
    """
    Create a styled scatter plot.
    
    Parameters:
    - df: pandas DataFrame
    - x: column name for x-axis
    - y: column name for y-axis
    - title: plot title
    """
    fig, ax = plt.subplots(figsize=(10.5, 6))
    
    # Background color
    ax.set_facecolor('#EAF4FB')
    
    # Titles and labels
    ax.set_title(title, fontsize=24.7)
    ax.set_xlabel(x, fontsize=20.8)
    ax.set_ylabel(y, fontsize=20.8)
    
    
    # Grid below points
    ax.set_axisbelow(True)
    ax.grid(True, which='major', color='#FFFFFF', linewidth=1)
    ax.grid(True, which='minor', color='#FFFFFF', linewidth=0.5)
    
    # Scatter points
    ax.scatter(
        df[x],
        df[y],
        color='#2980B9',
        alpha=0.1,
        s=60
    )
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.show()

# ---------------------------
# Functions
# ---------------------------

def plot_boxplot(
    df,
    x,
    y,
    hue,
    title,
    palette=category_colors,
    ax=None,
    show_legend=True
):
    """
    Create a styled boxplot and return fig, ax.
    
    If ax is provided, plot is drawn on it.
    If not, a new figure and axis are created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    else:
        fig = ax.figure

    # Background and grid
    ax.set_facecolor('#EAF4FB')
    ax.set_axisbelow(True)
    ax.grid(True, which='major', color='#FFFFFF', linewidth=1)
    ax.grid(True, which='minor', color='#FFFFFF', linewidth=0.5)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Boxplot
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        ax=ax
    )

    # Titles and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel(y, fontsize=14)

    if show_legend:
        ax.legend(title=hue, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.get_legend().remove()

    return fig, ax





def plot_histogram(
    df,
    x,
    title,
    bins=20,
    ax=None
):
    """
    Create a styled histogram and return fig, ax.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 6))
    else:
        fig = ax.figure

    ax.set_facecolor('#EAF4FB')
    ax.set_axisbelow(True)

    ax.grid(True, which='major', color='#FFFFFF', linewidth=1)
    ax.grid(True, which='minor', color='#FFFFFF', linewidth=0.5)

    ax.hist(
        df[x],
        bins=bins,
        color='#2980B9',
        alpha=0.7,
        edgecolor='white'
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax



def plot_scatter_subplots(df, x, y, title, color='#2980B9'):
    """
    Create side-by-side scatter plots for synchronous and asynchronous update modes.
    
    Parameters:
    - df: pandas DataFrame
    - x: column name for x-axis
    - y: column name for y-axis
    - title: main figure title
    - color: color for all points
    """
    # Unique update modes
    modes = ['synchronous', 'asynchronous']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    fig.suptitle(title, fontsize=24)
    
    for ax, mode in zip(axes, modes):
        # Filter dataframe for the update mode
        df_mode = df[df['update_mode'] == mode]
        
        # Background color
        ax.set_facecolor('#EAF4FB')
        ax.set_axisbelow(True)
        ax.grid(True, which='major', color='#FFFFFF', linewidth=1)
        ax.grid(True, which='minor', color='#FFFFFF', linewidth=0.5)
        
        # Scatter plot without hue
        ax.scatter(
            df_mode[x],
            df_mode[y],
            color=color,
            alpha=0.5,
            s=60
        )
        
        ax.set_title(f'{mode.capitalize()} updates', fontsize=18)
        ax.set_xlabel(x, fontsize=16)
        ax.set_ylabel(y, fontsize=16)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
