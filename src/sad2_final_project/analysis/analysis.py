import pandas as pd
from pathlib import Path
from typing import List


# --------------------------------------------------
# MAx
# --------------------------------------------------

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


# TODO - add function to add score functions to existing data 



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

def plot_boxplot(df, x, y, hue, title, palette=category_colors):
    """
    Create a styled boxplot for a given hue.
    
    Parameters:
    - df: pandas DataFrame
    - x: column name for x-axis
    - y: column name for y-axis
    - hue: column name for grouping
    - title: plot title
    - palette: list of colors
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    
    # Background color and grid below
    ax.set_facecolor('#EAF4FB')
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', color='#FFFFFF', linewidth=1)
    ax.yaxis.grid(True, which='major', color='#FFFFFF', linewidth=1)
    ax.xaxis.grid(True, which='minor', color='#FFFFFF', linewidth=0.5)
    ax.yaxis.grid(True, which='minor', color='#FFFFFF', linewidth=0.5)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Boxplot
    sns.boxplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax)
    
    # Titles and labels
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x, fontsize=18)
    ax.set_ylabel(y, fontsize=18)
    ax.legend(title=hue, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(df, x, title, bins=20):
    """
    Create a styled histogram.
    
    Parameters:
    - df: pandas DataFrame
    - x: column name for x-axis
    - title: plot title
    - bins: number of bins
    """
    fig, ax = plt.subplots(figsize=(10.5, 6))
    
    # Background color
    ax.set_facecolor('#EAF4FB')
    
    # Titles and labels
    ax.set_title(title, fontsize=24.7)
    ax.set_xlabel(x, fontsize=20.8)
    ax.set_ylabel("Count", fontsize=20.8)
    
    # Ticks
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    
    # Grid below bars
    ax.set_axisbelow(True)
    ax.grid(True, which='major', color='#FFFFFF', linewidth=1)
    ax.grid(True, which='minor', color='#FFFFFF', linewidth=0.5)
    
    # Histogram bars
    ax.hist(
        df[x],
        bins=bins,
        color='#2980B9',
        alpha=0.7,
        edgecolor='white'
    )

    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.show()

    from matplotlib.ticker import MultipleLocator
    import matplotlib.pyplot as plt