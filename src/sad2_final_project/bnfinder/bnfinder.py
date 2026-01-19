import os
import pandas as pd
from pathlib import Path
import sad2_final_project.bnfinder.bnfinder_wrapper as bnf
from sad2_final_project.bnfinder.score_functions import score_dag_from_sif

#TODO - check if necessary 
def _load_external_data(filepath):
    """
    Loads data from a CSV file.
    Assumes rows = time steps, columns = gene names.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"External data file not found: {filepath}")
    
    print(f"-> Loading external data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Loaded dataset with shape: {df.shape}")
    return df

def _load_ground_truth(filepath):
    """
    Loads ground truth edges from a CSV for evaluation.
    Expected format: Parent,Child
    """
    if not os.path.exists(filepath):
        return None
    
    print(f"-> Loading ground truth from: {filepath}")
    edges = set()
    df = pd.read_csv(filepath)
    for _, row in df.iterrows():
        edges.add((str(row[0]), str(row[1]))) # Parent, Child
    return edges

def _evaluate_results_metrics(true_edges, inferred_edges):
    """Calculates Precision/Recall if ground truth is available."""
    true_set = set(true_edges)
    inferred_set = set(inferred_edges)
    
    tp = len(true_set.intersection(inferred_set))
    fp = len(inferred_set - true_set)
    # tn to jest dopełnienie nie wpliczonych krawędzi_
    # n_nodes = trzeba jakoś ze struktury true_edges
    # total_possible = n_nodes * (n_nodes - 1)
    # tn = total_possible - tp - fp - fn
    fn = len(true_set - inferred_set)

    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ahd = fp + fn

    # MAX: I changed return value so i can generate csv files 
    return {
        "TP": tp,
        # "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "AHD": ahd
    }

# # TODO, zastąpiona przez 
# def _evalute_score_values():
#     pass


# TODO CHECK 
# TODO - przepisany main z oryginalne task3
def run_bnfinder(
    dataset_path: Path | str, # dataset path for learning 
    ground_truth_path: Path | str | None = None, # ground truth for metrics
    score_functions: list[str] = ["MDL", "BDE"],
    bnf_file_path: Path | str = f"model_1_bnf_formatted.txt", # path for bnf_format
    trained_model_name: Path | str = "model_1",  #TODO to jest folder do metryk, ale je trzeba jakoś przechwycić 
    metrics_file: Path | str = f'model_1_bnf_metric.csv' #TODO najlepiej ten sam co output 
):
    # Paths managements
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem
    # Data management
    ## 1. Load Data
    try:
        df = _load_external_data(dataset_path)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    ## 2. Convert to BNFinder Format
    ### This uses the function from 'bnfinder_wrapper.py' to handle the '#default 0 1' header
    bnf.write_bnf_input(df, bnf_file_path)

    ## 3. Load Ground Truth (if available)
    true_edges = _load_ground_truth(ground_truth_path)

    # Inference
    ## 4. Run Inference on given score functions (default = MDL & BDe)
    print("\n=== STARTING INFERENCE ===")
    rows = []
    for score in score_functions:
        output_sif = Path(str(trained_model_name) + f'_{score}.sif')  #
        
        try:
            ### Run wrapper
            bnf.run_bnfinder(bnf_file_path, output_sif, score=score)
            
            ### Parse output
            inferred_edges = bnf.parse_sif_results(output_sif)
            print(f"[{score}] Inferred {len(inferred_edges)} edges.")
            #TODO - problem, część plików jest nie obsługiwana 
            

            ### Metrics
            #### case 0 - no edges:
            if len(inferred_edges) == 0:
                print(f"[{score}] Empty graph — returning zero metrics")

                row = {
                    "dataset": dataset_name,
                    "score": score,
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "sensitivity": 0.0,
                    "AHD": 0,
                    "log_likelihood": 0.0,
                    "BIC": 0.0,
                    "MDL": 0.0,
                    "n_parameters": 0,
                }
                rows.append(row)
                continue
            #### case 1 - edges
            if true_edges is not None:
                print("   (Ground truth file found, evaluating metrics)")
                #### Obtain metrics
                metrics = _evaluate_results_metrics(true_edges, inferred_edges)
                #### obtain cost function: 
                cost_functions = score_dag_from_sif(dataset_df=df, sif_file_path=output_sif)
                #### Sanity check
                assert isinstance(metrics, dict)
                assert isinstance(cost_functions, dict)
                #### Format metrics
                row = {
                    ##### TODO - check i f overwriting work properly 
                    ##### groups
                    "dataset": dataset_name,
                    "score": score,
                    ##### metrics
                    "TP": metrics.get("TP"),
                    "FP": metrics.get("FP"),
                    "FN": metrics.get("FN"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "sensitivity": metrics.get("sensitivity"),
                    "AHD": metrics.get("AHD"),
                    ##### cost functions
                    "log_likelihood": cost_functions.get("log_likelihood"),
                    "MDL": cost_functions.get("MDL"),
                    "BDe": cost_functions.get("BDe"),
                    # "n_parameters": cost_functions.get("n_parameters"),
                }
                rows.append(row)
            #### case 3: no file
            else:
                print("   (No ground truth file found, skipping evaluation)")

        except Exception as e:
            print(f"[{score}] Failed: {e}")

    ## Append values to csv
    if rows:
        df_out = pd.DataFrame(rows)
        if os.path.exists(metrics_file):
            df_out.to_csv(metrics_file, mode="a", header=False, index=False)
        else:
            df_out.to_csv(metrics_file, index=False)

    print("\n=== DONE ===")