import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------
# MAx
# --------------------------------------------------


# --------------------
# Loading data
# --------------------



# CURRENT DATA
def loader_current_data(metadata_path: Path | str, results_path: Path | str) -> pd.DataFrame:
    '''
    Loads data in obsolete format into one dataframe
    '''
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
            
        if idx % 100:
          total = df_result.shape[0]
          print(f"[Progress] {idx}/{total} conditions completed ({100*idx/total:.1f}%)")
        
    return df_result

# TODO LIBRARY: add score functions to existing data



# --------------------
# Analysis 2
# --------------------

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

        if idx % 100:
          total = meta_df.shape[0]
          print(f"[Progress] {idx}/{total} conditions completed ({100*idx/total:.1f}%)")

    
    return meta_df.merge(pd.DataFrame(records), how='outer', on=index_column)




# --------------------------------------------------
# Joanna
# --------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
import seaborn as sns

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
# Global style
# ---------------------------
rcParams['font.family'] = 'Liberation Serif'
rcParams['font.size'] = 16
rcParams['text.color'] = '#2C3E50'

# example category colors
category_colors = ['#C0392B', '#2980B9', '#27AE60', '#F4D03F', '#8E44AD']

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
