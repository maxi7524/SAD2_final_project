import networkx as nx
from boolean import BooleanAlgebra
import random
import logging
from .bn import BN
import csv
import os
import random

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
    Generates a dataset from multiple trajectories and saves to CSV for BNFinder.
    Returns True if dataset accepted and saved, False if rejected.
    """

    min_ratio = max(0.0, target_attractor_ratio - tolerance)
    max_ratio = min(1.0, target_attractor_ratio + tolerance)
    # TODO REMOVE: debug mode - trajectories
    # print('Ratios')
    # print(min_ratio)
    # print(max_ratio)

    dataset_rows = []
    attractor_count = 0
    total_count = 0
    max_total_states = num_trajectories * trajectory_length

    for traj_index in range(num_trajectories):
        #
        trajectory, att_count, trans_count = bn_instance.simulate_trajectory(
            sampling_frequency=sampling_frequency,
            trajectory_length=trajectory_length
        )
        # print(att_count, trans_count)
        traj_total = att_count + trans_count
        attractor_count += att_count
        total_count += traj_total

        # TODO REMOVE: debug mode trajectories
        # dodajemy trajektorię do datasetu
        dataset_rows.append(trajectory)

        # --- wczesne odrzucenie ---
        remaining_states = max_total_states - total_count
        # print(f'{remaining_states=}')
        max_possible_ratio = (attractor_count + remaining_states) / max_total_states
        # print(f'{max_possible_ratio=}')
        min_possible_ratio = attractor_count / max_total_states
        # print(f'{min_possible_ratio=}')
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