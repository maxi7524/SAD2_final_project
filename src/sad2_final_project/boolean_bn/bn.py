import networkx as nx
from boolean import BooleanAlgebra
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from typing import Literal
import pandas as pd
import os


# ============================================================
# Boolean Network Class
# ============================================================

class BN():
    """
    Class for generating Boolean Networks which supports generating trajectories
    """
    __bool_algebra = BooleanAlgebra()

    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------

    def __int_to_state(self, x: int) -> tuple[int, ...]:
        """
        Helper method for converting a non-negative integer into a state in the form of a tuple of 0s and 1s.

            Args:
                x (int): A state number

            Returns:
                tuple[int, ...]: A tuple of 0s and 1s representing the Boolean network state.
        """
        binary_str = format(x,'0'+str(self.num_nodes)+'b') 
        state = [int(char) for char in binary_str]

        return tuple(state)

    def _state_to_expression(self, state: tuple[int, ...]) -> dict:
        """
        Converts a Boolean network state represented as 0s and 1s into a dictionary
        mapping each node to its corresponding BooleanAlgebra expression (TRUE or FALSE).
        """
        return {
            self.list_of_nodes[i]:
                self.__bool_algebra.TRUE if state[i] == 1
                else self.__bool_algebra.FALSE
            for i in range(self.num_nodes)
        }

    @staticmethod
    def __state_to_binary_str(state: tuple[int, ...]) -> str:
        """
        Converts a Boolean network state from a tuple of 0s and 1s into a binary string.

            Args:
                state (tuple[int, ...]): A tuple of 0s and 1s representing the Boolean network state

            Returns:
                str: A binary string representing the Boolean network state
        """
        bin_str = ''
        for bit in state:
            bin_str += str(bit)
        
        return bin_str

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------

    def __init__(self, num_nodes: int, mode: Literal["synchronous", "asynchronous"], functions: list = None, *, n_parents_per_node=[2,3]):
        """
        Initialize a Boolean network.

        Args:
            num_nodes (int): Number of nodes in the network.
            mode (str): Update mode ("synchronous" or "asynchronous").
            functions (list, optional): Boolean update functions. If None,
                random functions are generated.
            n_parents_per_node (list[int], optional): Possible numbers of parents
                per node when generating random functions.
        """

        self.mode = mode # mode in which trajectories will be generated
        self.num_nodes = num_nodes # number of nodes in boolean network
        self.node_names = [f"x{i}" for i in range(num_nodes)]
        self.TF = {True: 1, False: 0} # helper dictionary for conversion True and False into 1 or 0

        self.list_of_nodes = [] # List of BooleanAlgebra.Symbol objects representing each node
        for node_name in self.node_names:
            node = self.__bool_algebra.Symbol(node_name)
            self.list_of_nodes.append(node)

        # Initialize Boolean functions: use provided ones if given, otherwise generate randomly
        if functions is not None:
            if len(functions) != self.num_nodes:
                raise ValueError("Number of functions must match number of nodes.")
            self.functions = functions
        else:
            self.functions = self.generate_random_functions(n_parents_per_node=n_parents_per_node)

        # Precompute attractors
        self.attractors = self.get_attractors()

    # ------------------------------------------------------------
    # State transition logic
    # ------------------------------------------------------------

    def _get_neighbor_states(self, state: tuple[int, ...]) -> set[tuple[int, ...]]:
        """
        Computes the states reachable from the given state in one step of update.
        """
        if self.mode == "synchronous":
            # In synchronous mode, only one next state is possible
            return {self._next_synchronous(state)}
        else:
            # In asynchronous mode, each transition updates a single node
            reachable_states = [
                self._next_asynchronous(state, i)
                for i in range(self.num_nodes)
            ]
            return set(reachable_states)

    def _generate_state_transition_system(self)-> nx.DiGraph:
        """
        Generates state transition system of the Boolean network. Works for both synchronous and asynchronous.
        """

        G = nx.DiGraph()

        # adding nodes
        for n in range(2**self.num_nodes):
            node = self.__int_to_state(n) # transform x into binary representation (during iteration we get every possible state)
            G.add_node(node)

        # adding edges
        for node in G.nodes:
            reachable_states = self._get_neighbor_states(node)
            for reachable in reachable_states:
                G.add_edge(node, reachable)

        return G

    def _next_synchronous(self, curr_state: tuple[int, ...]) -> tuple[int, ...]:
        """
        Returns the next state under synchronous update.
        x(t+1) = (f1, f2,...,fn)
        """

        # Map integer state values to BooleanAlgebra expressions for evaluation
        vals = {
            self.list_of_nodes[i]:
                self.__bool_algebra.TRUE if curr_state[i] == 1
                else self.__bool_algebra.FALSE
            for i in range(self.num_nodes)
        }

        # x(t+1) = (f1, f2,...,fn) evaluates each function on corresponding coordinate
        try:
            new_state = [self.TF[func.subs(vals).simplify()] for func in self.functions]
        except Exception as e:
            raise

        return tuple(new_state)

    def _next_asynchronous(self, curr_state: tuple[int, ...], coordinate) -> tuple[int, ...]:
        """
        Returns the next state under asynchronous update.
        if given coordinate to update xj, updates: x(t+1) = x(x1, x2,...fi,...fn)
        if it is not given, it is drawn randomly
        """
        # conversion of ones and zeros into True or False (needed during calling subs)
        vals = {
            self.list_of_nodes[i]:
                self.__bool_algebra.TRUE if curr_state[i] == 1
                else self.__bool_algebra.FALSE
            for i in range(self.num_nodes)
        }

        # setting coordinate to evaluate function on
        i = coordinate if coordinate is not None else random.randint(0, self.num_nodes - 1)
        
        curr_state = list(curr_state)

        try:
            curr_state[i] = self.TF[self.functions[i].subs(vals).simplify()]
        except Exception as e:
            raise
        
        return tuple(curr_state)
            
    # ------------------------------------------------------------
    # Attractors
    # ------------------------------------------------------------

    def get_attractors(self) -> list[set[tuple[int]]]:
        """
        Computes the asynchronous attractors of the Boolean network.

            Returns:
                list[set[tuple[int]]]: A list of asynchronous attractors. Each attractor is a set of states.
        """
        sts = self._generate_state_transition_system()

        attractors = []
        try:
            for attractor in nx.attracting_components(sts):
                attractors.append(attractor)
        except Exception as e:
            raise

        return attractors

    # ------------------------------------------------------------
    # Random Boolean function generation
    # ------------------------------------------------------------

    def _build_random_boolean_expression(self, parents: list):
        """
        Builds a BooleanAlgebra expression from a list of parent symbols
        using randomly chosen binary operators (&, |).

        parents: list of BooleanAlgebra.Symbol (possibly negated)
        """
        assert len(parents) >= 1

        # single parent → identity
        if len(parents) == 1:
            return parents[0]

        expr = parents[0]

        for next_parent in parents[1:]:
            op = random.choice(["&", "|"])
            expr_str = f"({expr}) {op} ({next_parent})"
            try:
                expr = self.__bool_algebra.parse(expr_str)
            except Exception as e:
                raise

        return expr

    def generate_random_functions(self, n_parents_per_node: list[int] = [2, 3]):
        """
        Chooses parents and generates random Boolean functions.
        Supports arbitrary number of parents >= 1.
        """

        # choose number of parents per node
        num_parents_list = [
            random.choice(n_parents_per_node)
            for _ in range(self.num_nodes)
        ]

        functions = []

        def maybe_negate(var):
            return ~var if random.choice([True, False]) else var

        for num_parents in num_parents_list:

            if num_parents < 1:
                raise ValueError("Each node must have at least one parent.")

            if num_parents > self.num_nodes:
                raise ValueError(
                    f"Cannot sample {num_parents} parents from {self.num_nodes} nodes."
                )

            # sample parents
            parents = random.sample(self.list_of_nodes, num_parents)

            # optional negation
            parents = [maybe_negate(p) for p in parents]

            # build expression
            expr = self._build_random_boolean_expression(parents)
            functions.append(expr)

        return functions
    
    # ------------------------------------------------------------
    # Trajectory simulation
    # ------------------------------------------------------------

    def simulate_trajectory(
        self,
        sampling_frequency: int = 3,
        trajectory_length: int = 50
    ):
            """
            Simulates a trajectory of the Boolean network.
            """

            trajectory = []
            attractor_counter = 0
            transient_counter = 0

            total_steps = trajectory_length * sampling_frequency

            # random initial state
            last_state = tuple(random.choice([0, 1]) for _ in range(self.num_nodes))
            trajectory.append(last_state)
            
            # Count initial state
            is_attractor = any(last_state in attr for attr in self.attractors)
            if is_attractor:
                attractor_counter += 1
            else:
                transient_counter += 1

            for step in range(1, total_steps):

                if self.mode == "synchronous":
                    next_state = self._next_synchronous(last_state)
                else:
                    coordinate = random.randint(0, self.num_nodes - 1)
                    next_state = self._next_asynchronous(last_state, coordinate)

                last_state = next_state

                # Check whether state belongs to an attractor
                is_attractor = any(next_state in attr for attr in self.attractors)

                if is_attractor:
                    attractor_counter += 1
                else:
                    transient_counter += 1

                # sampling
                if step % sampling_frequency == 0:
                    trajectory.append(next_state)

                    if len(trajectory) == trajectory_length:
                        
                        return trajectory, attractor_counter, transient_counter

            # fallback (should not happen, but safe)
            return trajectory, attractor_counter, transient_counter

    # ------------------------------------------------------------
    # Saving and visualization
    # ------------------------------------------------------------

    def save_ground_truth(self, filepath: str = "ground_truth_edges.csv", with_header: bool = False) -> None:
        """
        Saves the ground truth edges (Parent, Child) to a CSV file based on the
        network's Boolean functions.

        Args:
            filepath (str): Destination path for the CSV file.
        """
        

        edges = []

        for i, func in enumerate(self.functions):
            child_name = self.node_names[i]
            parents = []
            if hasattr(func, 'get_symbols'):
                parents = func.get_symbols()
            elif func in [self.__bool_algebra.TRUE, self.__bool_algebra.FALSE]:
                parents = []

            for parent_symbol in parents:
                parent_name = str(parent_symbol)
                edges.append((parent_name, child_name))

        try:
            df = pd.DataFrame(edges, columns=["Parent", "Child"])
            if with_header:
                df.to_csv(filepath, index=False)
            else:
                df.to_csv(filepath, index=False, header=False)

            
        except Exception as e:
            raise

    def draw_state_transition_system(self, highlight_attractors: bool = True) -> None:
        """
        Draws the state transition system.

            Args:
                highlight_attractors: If True, states belonging to different attractors are drawn 
                    using distinct colors.

            Returns:
                None
        """
        

        # The color used for non-attractor states in the state transition system
        NON_ATTRACTOR_STATE_COLOR = 'lightgrey'

        try:
            sts = self._generate_state_transition_system()
        except Exception:
            raise

        if highlight_attractors:
            attractors = self.get_attractors()

            sts_nodes = list(sts.nodes)

            node_colors = [NON_ATTRACTOR_STATE_COLOR for node in sts_nodes]

            colors = list(mcolors.CSS4_COLORS)
            colors.remove('white')
            colors.remove(NON_ATTRACTOR_STATE_COLOR)
            
            for attractor in attractors:
                # Select a random color for coloring the states of the attractor
                color = random.choice(colors)
                for state in attractor:
                    node_colors[sts_nodes.index(state)] = color

        try:
            plt.figure(figsize=(12, 12))

            pos = nx.spring_layout(sts, seed=42, k=1.2)

            nx.draw_networkx_nodes(
                sts,
                pos,
                node_color=node_colors,
                node_size=3500,      
                edgecolors='black',
                linewidths=1.2,
                alpha=1.0
            )

            nx.draw_networkx_edges(
                sts,
                pos,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=20,
                width=1.5,
                alpha=0.7,
                connectionstyle='arc3,rad=0.15',
                min_source_margin=15,
                min_target_margin=25
            )

            nx.draw_networkx_labels(
                sts,
                pos,
                font_size=9,
                font_weight='bold'
            )

            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            raise

# ============================================================
# Utility Functions
# ============================================================

def load_bnet_to_BN(file_path: str) -> BN:
    """
    Parses a .bnet file and returns a BN object with functions converted to DNF.
    """
    nodes = []
    functions = []

    algebra = BooleanAlgebra()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#') or line.lower().strip() == 'targets,factors':
                continue

            target, expression = [part.strip() for part in line.split(',', 1)]
            parsed_expr = algebra.parse(expression)
            dnf_expr = algebra.dnf(parsed_expr)

            nodes.append(target)
            functions.append(str(dnf_expr))

    return BN(nodes, functions)



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


    tmp = load_bnet_to_BN("/Users/chmuradin/Desktop/tmp_sad/SAD2_final_project/model_id_37/model.bnet")

    # Create BN with fixed functions
    #bn = BN(num_nodes=3, mode="asynchronous", functions=functions)
    print(tmp.functions)

