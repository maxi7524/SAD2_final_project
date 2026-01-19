import networkx as nx
from boolean import BooleanAlgebra
import random
import logging
from .bn import BN

def save_trajectories_to_bnfinder_format(
        bn_instance,
        num_trajectories,
        output_file="trajectory.txt"
):
    """
    Simulates multiple trajectories and saves them to a file formatted for BNfinder.

    Args:
        bn_instance (BN): An instance of the BN class.
        num_trajectories (int): Number of independent trajectories to simulate.
        output_file (str): Name of the output file.
    """

    all_data = {name: [] for name in bn_instance.node_names}
    experiment_names = []

    for traj_idx in range(1, num_trajectories + 1):
        trajectory, _, _ = bn_instance.simulate_trajectory()

        for time_step in range(len(trajectory)):
            experiment_names.append(f"seq{traj_idx}:{time_step}")

        for state in trajectory:
            for node_idx, node_name in enumerate(bn_instance.node_names):
                val = state[node_idx]
                all_data[node_name].append(str(val))


    with open(output_file, 'w') as f:

        header = "conditions\t" + "\t".join(experiment_names) + "\n"
        f.write(header)

        for node_name in bn_instance.node_names:
            values_str = "\t".join(all_data[node_name])
            row = f"{node_name}\t{values_str}\n"
            f.write(row)

    print("Done.")


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