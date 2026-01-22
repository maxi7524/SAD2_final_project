import networkx as nx
from boolean import BooleanAlgebra
from .bn import BN
import csv

# ============================================================
# Dataset Generation Function
# ============================================================

def simulate_trajectories_to_csv(
        bn_instance,
        num_trajectories,
        output_file,
        sampling_frequency,
        trajectory_length,
        target_attractor_ratio,
        tolerance
        ):
    """
    Generate a dataset from multiple trajectories and save it to CSV for BNFinder.

    Returns:
        (bool, float): 
            - True if dataset meets the attractor ratio criteria and is accepted.
            - False otherwise.
            - The actual attractor ratio.
    """

    min_ratio = max(0.0, target_attractor_ratio - tolerance)
    max_ratio = min(1.0, target_attractor_ratio + tolerance)

    dataset_rows = []
    attractor_count = 0
    total_count = 0

    # Attempt dataset generation up to 2 times
    for _ in range(3):
        for traj_index in range(num_trajectories):

            # Simulate a single trajectory
            trajectory, att_count, trans_count = bn_instance.simulate_trajectory(
                sampling_frequency=sampling_frequency,
                trajectory_length=trajectory_length
            )

            traj_total = att_count + trans_count
            attractor_count += att_count
            total_count += traj_total

            dataset_rows.append(trajectory)
        
        # Compute final attractor ratio
        actual_ratio = attractor_count / total_count if total_count > 0 else 0.0
        if actual_ratio>=min_ratio and actual_ratio<=max_ratio:
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)

                # Write header (gene names)
                header = [f"G{i+1}" for i in range(len(dataset_rows[0][0]))]
                writer.writerow(header)

                # Write each trajectory, separated by an empty line
                for trajectory in dataset_rows:
                    writer.writerows(trajectory)
                    writer.writerow([]) 

            print(f"Dataset saved to {output_file} (actual attractor ratio: {actual_ratio:.2f})")
            return True, actual_ratio
        
    # Save the dataset even if it does not meet the ratio criteria
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        header = [f"G{i+1}" for i in range(len(dataset_rows[0][0]))]
        writer.writerow(header)

        for trajectory in dataset_rows:
            writer.writerows(trajectory)
            writer.writerow([])  

    print(f"Dataset saved to {output_file} (actual attractor ratio: {actual_ratio:.2f})")
    return False, actual_ratio

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    algebra = BooleanAlgebra()
    x1 = algebra.Symbol('x0')
    x2 = algebra.Symbol('x1')
    x3 = algebra.Symbol('x2')

    # Define functions (the same as in exercise in lab 4)
    f1 = x2
    f2 = ~x2
    f3 = ~x2 | x3

    functions = [f1, f2, f3]
    bn = BN(
        num_nodes=3,
        mode="asynchronous",
        trajectory_length=10,
        functions=functions
    )
