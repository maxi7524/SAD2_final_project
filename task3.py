import os
import pandas as pd
import bnfinder_wrapper as bnf  # Using the wrapper we created earlier

# --- CONFIGURATION ---
# Simulation data file path
EXTERNAL_DATA_PATH = "simulation_output.csv" 

# Where to define the "Ground Truth" for evaluation (optional)
# If this file doesn't exist, the script will skip evaluation.
GROUND_TRUTH_PATH = "ground_truth_edges.csv"

# Output filenames
BNF_INPUT_FILE = "bnf_formatted_input.txt"
OUTPUT_TEMPLATE = "inferred_network_{}.sif"

def load_external_data(filepath):
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

def load_ground_truth(filepath):
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

def evaluate_results(true_edges, inferred_edges):
    """Calculates Precision/Recall if ground truth is available."""
    true_set = set(true_edges)
    inferred_set = set(inferred_edges)
    
    tp = len(true_set.intersection(inferred_set))
    fp = len(inferred_set - true_set)
    fn = len(true_set - inferred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return tp, fp, fn, precision, recall


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    try:
        df = load_external_data(EXTERNAL_DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # 2. Convert to BNFinder Format
    # This uses the function from 'bnfinder_wrapper.py' to handle the '#default 0 1' header
    bnf.write_bnf_input(df, BNF_INPUT_FILE)

    # 3. Load Ground Truth (if available)
    true_edges = load_ground_truth(GROUND_TRUTH_PATH)

    # 4. Run Inference (MDL & BDe)
    scorers = ["MDL", "BDE"]
    
    print("\n=== STARTING INFERENCE ===")
    
    for score in scorers:
        output_sif = OUTPUT_TEMPLATE.format(score)
        
        try:
            # Run wrapper
            bnf.run_bnfinder(BNF_INPUT_FILE, output_sif, score=score)
            
            # Parse output
            inferred_edges = bnf.parse_sif_results(output_sif)
            print(f"[{score}] Inferred {len(inferred_edges)} edges.")
            
            # Evaluation
            if true_edges is not None:
                tp, fp, fn, prec, rec = evaluate_results(true_edges, inferred_edges)
                print(f"   Evaluation: TP={tp}, FP={fp}, FN={fn}")
                print(f"   Metrics:    Precision={prec:.2f}, Recall={rec:.2f}")
            else:
                print("   (No ground truth file found, skipping evaluation)")
                
        except Exception as e:
            print(f"[{score}] Failed: {e}")

    print("\n=== DONE ===")