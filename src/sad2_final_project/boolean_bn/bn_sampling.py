import networkx as nx
from boolean import BooleanAlgebra
import random
import logging
from .bn import BN
import csv
import os
import random

# def save_trajectories_to_bnfinder_format(
#         bn_instance,
#         num_trajectories,
#         output_file="trajectory.txt"
# ):
#     """
#     Simulates multiple trajectories and saves them to a file formatted for BNfinder.

#     Args:
#         bn_instance (BN): An instance of the BN class.
#         num_trajectories (int): Number of independent trajectories to simulate.
#         output_file (str): Name of the output file.
#     """

#     all_data = {name: [] for name in bn_instance.node_names}
#     experiment_names = []

#     for traj_idx in range(1, num_trajectories + 1):
#         trajectory, _, _ = bn_instance.simulate_trajectory()

#         for time_step in range(len(trajectory)):
#             experiment_names.append(f"seq{traj_idx}:{time_step}")

#         for state in trajectory:
#             for node_idx, node_name in enumerate(bn_instance.node_names):
#                 val = state[node_idx]
#                 all_data[node_name].append(str(val))


#     with open(output_file, 'w') as f:

#         header = "conditions\t" + "\t".join(experiment_names) + "\n"
#         f.write(header)

#         for node_name in bn_instance.node_names:
#             values_str = "\t".join(all_data[node_name])
#             row = f"{node_name}\t{values_str}\n"
#             f.write(row)

#     print("Done.")


# def simulate_trajectories_to_csv(
#         bn_instance,
#         num_trajectories,
#         output_file="trajectory.csv",
#         *, 
#         sampling_frequency: int = 3,
#         target_attractor_ratio: float = 0.4, # Approximate fraction of trajectory in attractor (0-1)
#         tolerance: float = 0.1, # Allowed deviation from the calculated entrance step (0-1)
#         max_iter: int = 50, # Maximum attempts to generate a valid state per step before restarting
#         max_trajectory_restarts: int = 100 # Maximum number of trajectory restarts allowed
# ):
#     rows = []

#     for _ in range(num_trajectories):
#         trajectory, _, _ = bn_instance.simulate_trajectory(
#             sampling_frequency = sampling_frequency,
#             target_attractor_ratio = target_attractor_ratio, # Approximate fraction of trajectory in attractor (0-1)
#             tolerance = tolerance, # Allowed deviation from the calculated entrance step (0-1)
#             max_iter = max_iter, # Maximum attempts to generate a valid state per step before restarting
#             max_trajectory_restarts= max_trajectory_restarts # Maximum number of trajectory restarts allowed
#         )

#         for state in trajectory:
#             rows.append(state)  # zostawiamy wartości jako int/bool

#     with open(output_file, "w", newline="") as f:
#         writer = csv.writer(f)

#         # nagłówki (G1, G2, G3, ...)
#         header = [f"G{i+1}" for i in range(len(rows[0]))]
#         writer.writerow(header)

#         # dane
#         writer.writerows(rows)

#     print("Done.")

# num_nodes: int, mode: Literal["synchronous", "asynchronous"], functions: list = None, *, n_parents_per_node=[2,3]


##### tutaj generujemy jeden dataset jeżeli w pewnym momencie nie zgadza nam sie liczba stanów attraktorowych zwracamyu null jeżeli uda sie spełnić wymagania to zapisujemuy do csv
def simulate_dataset_to_csv(
    output_file: str,
    target_attractor_ratio: float,
    number_of_trajectories: int,
    tolerance: float = 0.1,
    sampling_frequency: int = 3,
    trajectory_length: int = 50,
):
    """
    Generates a dataset from multiple trajectories and saves to CSV for BNFinder.
    Returns True if dataset accepted and saved, False if rejected.
    """
    # losujemy liczbę wierzchołków BN
    num_nodes = random.randint(5, 16)

    # losujemy tryb działania BN
    mode = random.choice(["synchronous", "asynchronous"])

    # tworzymy nową instancję BN
    bn_instance = BN(
        num_nodes=num_nodes,
        mode=mode
    )
    min_ratio = max(0.0, target_attractor_ratio - tolerance)
    max_ratio = min(1.0, target_attractor_ratio + tolerance)

    dataset_rows = []
    attractor_count = 0
    total_count = 0
    max_total_states = number_of_trajectories * trajectory_length

    for traj_index in range(number_of_trajectories):
        trajectory, att_count, trans_count = bn_instance.simulate_trajectory(
            sampling_frequency=sampling_frequency,
            trajectory_length=trajectory_length
        )

        traj_total = att_count + trans_count
        attractor_count += att_count
        total_count += traj_total

        # dodajemy trajektorię do datasetu
        dataset_rows.append(trajectory)

        # --- wczesne odrzucenie ---
        remaining_states = max_total_states - total_count
        max_possible_ratio = (attractor_count + remaining_states) / max_total_states
        min_possible_ratio = attractor_count / max_total_states

        if max_possible_ratio < min_ratio or min_possible_ratio > max_ratio:
            return False  # dataset odrzucony

    # finalna proporcja
    actual_ratio = attractor_count / total_count if total_count > 0 else 0.0

    if not (min_ratio <= actual_ratio <= max_ratio):
        return False  # dataset odrzucony

    # --- zapis do CSV ---
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # nagłówki
        header = [f"G{i+1}" for i in range(len(dataset_rows[0][0]))]
        writer.writerow(header)

        # zapis każdej trajektorii z pustą linią między nimi
        for trajectory in dataset_rows:
            writer.writerows(trajectory)
            writer.writerow([])  # linia przerwy

    print(f"Dataset saved to {output_file} (actual attractor ratio: {actual_ratio:.2f})")
    return True


# generuje wiele datasetów, sama losuje im parametry jeżeli chcemy je podawać to trzeba to zmienić 
# zwraca ile datasetóœ spełniło wymagania
def simulate_random_varied_datasets_to_csv(
    output_dir: str,
    num_datasets: int = 10,
    target_ratio_range: tuple = (0.2, 0.8),
    sampling_frequency_range: tuple = (1, 5),
    number_of_trajectories_range: tuple = (5, 15),
    trajectory_length_range: tuple = (50, 150),
    tolerance: float = 0.1
):
    """
    Generates multiple datasets with randomly chosen parameters within given ranges.
    Saves accepted datasets to CSV using `simulate_dataset_to_csv`.
    Returns stats: number of accepted and rejected datasets.
    """

    os.makedirs(output_dir, exist_ok=True)
    stats = {"accepted": 0, "rejected": 0}

    for idx in range(num_datasets):
        # losujemy parametry
        ratio = random.uniform(*target_ratio_range)
        freq = random.randint(*sampling_frequency_range)
        n_traj = random.randint(*number_of_trajectories_range)
        traj_len = random.randint(*trajectory_length_range)

        filename = os.path.join(output_dir, f"dataset_{idx+1}_ratio_{ratio:.2f}.csv")

        success = simulate_dataset_to_csv(
            output_file=filename,
            target_attractor_ratio=ratio,
            number_of_trajectories=n_traj,
            tolerance=tolerance,
            sampling_frequency=freq,
            trajectory_length=traj_len
        )

        if success:
            stats["accepted"] += 1
        else:
            stats["rejected"] += 1

    print(f"Finished generating datasets. Accepted: {stats['accepted']}, Rejected: {stats['rejected']}")
    return stats



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

    # # 3. Generate and Save
    # save_trajectories_to_bnfinder_format(bn, 10, "bnfinder_input_test.txt")