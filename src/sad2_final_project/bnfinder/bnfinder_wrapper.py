import subprocess
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def get_bnfinder_path():
    """
    Automatically locates the 'bnf' executable.
    Prioritizes the system PATH, then checks the standard user install location.
    """
    # 1. Try to find 'bnf' in the system PATH
    path = shutil.which("bnf")
    
    # 2. If not found, check the standard user install location (~/.local/bin)
    if path is None:
        user_path = os.path.expanduser("~/.local/bin/bnf")
        if os.path.exists(user_path):
            path = user_path

    if path is None:
        raise FileNotFoundError(
            "Could not find 'bnf' executable. \n"
            "Please ensure it is installed via 'python2 -m pip install BNfinder' "
            "and that ~/.local/bin is in your PATH."
        )
    
    return path

def write_bnf_input(data_df, filename="experiment_data.txt"):
    """
    Writes a Pandas DataFrame to BNFinder input format.
    
    Args:
        data_df (pd.DataFrame): DataFrame where rows are time steps and columns are genes.
        filename (str): The name of the output text file.
    """
    with open(filename, 'w') as f:
        # 1. PREAMBLE
        # The '#default 0 1' directive tells BNFinder that all unspecified variables 
        # (which is all of them here) are discrete Boolean variables.
        f.write("#default 0 1\n")
        
        # 2. EXPERIMENT SPECIFICATION
        # For Dynamic Bayesian Networks, we define time points as SeriesName:Index.
        # We assume the dataframe represents a single continuous time series 'S1'.
        headers = [f"S1:{i+1}" for i in range(len(data_df))]
        
        # 'dataset1' is the label for this experiment set
        f.write("dataset1 " + " ".join(headers) + "\n")
        
        # 3. DATA
        # BNFinder expects genes as rows and experiments/time-points as columns.
        # We assume data_df has genes as columns, so we iterate over columns.
        for gene in data_df.columns:
            # Convert numeric values to strings
            values = data_df[gene].astype(str).tolist()
            f.write(f"{gene} " + " ".join(values) + "\n")

    print(f"-> Data written to {filename}")

def run_bnfinder(input_file, output_sif, score="MDL"):
    """
    Wrapper to call the legacy python2 BNfinder script via subprocess.
    
    Args:
        input_file (str): Path to the generated .txt input file.
        output_sif (str): Path where the result .sif file should be saved.
        score (str): Scoring criterion. Must be 'MDL', 'BDE', or 'MIT'.
    """
    bnf_path = get_bnfinder_path()
    
    # Construct command: python2 path/to/bnf -e input -n output -s Score -v
    # Note: We use lowercase '-s' based on the installed version 2.1.1
    # TODO remove suffix 
    
    cmd = [
        "python2", 
        bnf_path,
        "-e", input_file,
        "-n", output_sif,
        "-s", score,
        "-v"
    ]

    print(f"-> Running BNFinder with score: {score}...")
    
    try:
        # Check=True raises an error if the return code is non-zero
        # We capture stdout/stderr to keep the main terminal clean, 
        # but you can print them for debugging if needed.
        
        #TODO add proper path 
        log_path = Path(str(output_sif.parent) + '/' + output_sif.stem + f'_log.txt')
        # add proper path 
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd, 
                check=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True
            )
        print(f"   Success! Output saved to {output_sif}")
        #TODO - funkcja do odczytu:
        # musimy przechwycić wynik, i jakoś go zapisać w innym pliku, jako score 

        
    except subprocess.CalledProcessError as e:
        print("   BNFinder execution failed!")
        print("   Error Output:\n", e.stderr)
        raise e
#### split nie działało dla tuple
# def parse_sif_results(sif_file):
#     """
#     Parses the SIF output file to extract inferred edges.
    
#     Args:
#         sif_file (str): Path to the .sif file.
        
#     Returns:
#         list of tuples: [(parent, child), ...]
#     """
#     edges = []
#     if not os.path.exists(sif_file):
#         print(f"   Warning: SIF file {sif_file} not found (possibly no edges inferred).")
#         return edges

#     with open(sif_file, 'r') as f:
#         for line in f:
#             # TODO:  wysypuje sie pokazuje, że to tuple
#             parts = line.strip().split()
#             print(parts)
#             # SIF format: Parent Label Child (e.g., G1 + G2)
#             if len(parts) >= 3:
#                 parent = parts[0]
#                 child = parts[2]
#                 edges.append((parent, child))
#     return edges

def parse_sif_results(sif_file):
    """
    Parses the SIF output file to extract inferred edges.
    
    Args:
        sif_file (str or Path): Path to the .sif file.
        
    Returns:
        list of tuples: [(parent, child), ...]
    """
    edges = []
    if not os.path.exists(sif_file):
        print(f"   Warning: SIF file {sif_file} not found (possibly no edges inferred).")
        return edges

    with open(sif_file, 'r') as f:
        for line in f:
            # upewniamy się, że mamy string
            # if isinstance(line, (list, tuple)):
            #     parts = line
            # else:
            #     parts = line.strip().split()
            
            if len(line) >= 3:
                print(3)
                parent = line[0]
                child = line[2]
                edges.append((parent, child))
    
    return edges

def load_structure_from_sif(sif_path: str) -> dict[str, list[str]]:
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

# --- MAIN EXECUTION (Task 3 Logic) ---
if __name__ == "__main__":
    print("--- Setting up Task 3 Simulation ---")
    
    # 1. Generate Dummy Data (Simulating Part 1 output)
    # We create a simple causal chain: G1(t) -> G2(t+1)
    # This allows us to verify if the tool is actually working.
    np.random.seed(42)
    data_length = 50
    
    g1 = np.random.randint(0, 2, size=data_length)
    g2 = np.roll(g1, 1) # G2 is G1 shifted by 1 step (G1 causes G2)
    g3 = np.random.randint(0, 2, size=data_length) # G3 is pure noise
    
    # Create DataFrame (Genes as columns)
    df = pd.DataFrame({'G1': g1, 'G2': g2, 'G3': g3})
    
    # 2. Write Input File
    input_filename = "task3_test_input.txt"
    write_bnf_input(df, input_filename)

    # 3. Run Inference with required scorers: MDL and BDe
    # Note: BNFinder might be case-sensitive depending on version, 
    # but usually 'MDL' and 'BDE' (or 'BDe') are standard.
    scorers = ["MDL", "BDE"]
    inference_results = {}

    print("\n--- Starting Inference ---")
    for score in scorers:
        output_filename = f"network_{score}.sif"
        
        try:
            run_bnfinder(input_filename, output_filename, score=score)
            edges = parse_sif_results(output_filename)
            inference_results[score] = edges
        except Exception as e:
            print(f"   Skipping {score} due to error.")

    # 4. Display Comparison Results
    print("\n--- Task 3 Results Summary ---")
    print(f"True Relation: G1 -> G2")
    
    for score, edges in inference_results.items():
        print(f"\nScoring Function: {score}")
        if not edges:
            print("  Result: Empty Network (No edges found)")
        else:
            for parent, child in edges:
                match_status = "CORRECT" if parent == "G1" and child == "G2" else "FALSE POSITIVE"
                print(f"  Result: {parent} -> {child} [{match_status}]")