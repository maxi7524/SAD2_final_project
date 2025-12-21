import networkx as nx
from boolean import BooleanAlgebra
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from typing import Literal

class BN():
    __bool_algebra = BooleanAlgebra()


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

    # added
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


    def __init__(self, num_nodes: int, mode: Literal["synchronous", "asynchronous"]):
        """
        Class constructor
        """
        self.num_nodes = num_nodes
        self.node_names = [f"x{i}" for i in range(num_nodes)]
        self.TF = {True: 1, False: 0} # helper dictionary for conversion True and False into 1 or 0

        self.list_of_nodes = [] # List of BooleanAlgebra.Symbol objects representing each node
        for node_name in self.node_names:
            node = self.__bool_algebra.Symbol(node_name)
            self.list_of_nodes.append(node)
        
        self.mode = mode
        
        # Initialize Boolean functions and compute attractors   
        self.functions = self.generate_random_functions()
        self.attractors = self.get_attractors()

    # added 
    def get_neighbor_states(self, state: tuple[int, ...]) -> set[tuple[int, ...]]:
        """
        Computes the states reachable from the given state in one step of update.
        """
        if self.mode == "synchronous":
            # In synchronous mode, only one next state is possible
            return {self.next_synchronous(state)}
        else:
            # In asynchronous mode, each transition updates a single node
            reachable_states = [
                self.next_asynchronous(state, i)
                for i in range(self.num_nodes)
            ]
            return set(reachable_states)

    # added
    def generate_state_transition_system(self)-> nx.DiGraph:
        """
        Generates state transition system of the Boolean network. Works for both synchronous an asynchronous.
        """
        G = nx.DiGraph()

        # adding nodes
        for n in range(2**self.num_nodes):
            node = self.__int_to_state(n) # transorm x into binary representation (during iteration we get every possible state)
            G.add_node(node)

        # adding edges
        for node in G.nodes:
            reachable_states = self.get_neighbor_states(node)
            for reachable in reachable_states:
                G.add_edge(node, reachable)

        return G

    # added
    def next_synchronous(self, curr_state: tuple[int, ...]) -> tuple[int, ...]:
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
        new_state = [self.TF[func.subs(vals).simplify()] for func in self.functions]
        return tuple(new_state)

    # added
    def next_asynchronous(self, curr_state: tuple[int, ...], coordinate) -> tuple[int, ...]:
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
        curr_state[i] = self.TF[self.functions[i].subs(vals).simplify()]
        
        return tuple(curr_state)
            
    def get_attractors(self) -> list[set[tuple[int]]]:
        """
        Computes the asynchronous attractors of the Boolean network.

            Returns:
                list[set[tuple[int]]]: A list of asynchronous attractors. Each attractor is a set of states.
        """
        sts = self.generate_state_transition_system()

        attractors = []
        for attractor in nx.attracting_components(sts):
            attractors.append(attractor)

        return attractors
    
    # Not yet fully implemented; current version is for testing purposes
    def generate_random_functions(self):
        """
        Fixed Boolean functions for 3 nodes (x0, x1, x2) for testing.
        """

        if self.num_nodes != 3:
            raise ValueError("This test implementation supports exactly 3 nodes.")

        x0, x1, x2 = self.list_of_nodes

        functions = [
            (x0 & ~x1) | x2,     # f0
            x0 & x2,             # f1 
            (x1 | x2) & ~x0      # f2
        ]

        return functions

    def simulate_trajectories(self, start_state: tuple[int, ...], sampling_frequency: int, mode: str):
      """
      simulates trajectories of boolean network
      """
      pass
    
    def scoring_function(self, method: Literal['MDL', 'DBE']):
        """
        scoring function for graphs
        """
    
    def evaluate_accuracy(self):
        """
        Evaluate accuracy of the reconstructed network
        Share parameters with the function above e.g. move methodto init?
        """

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
        NON_ATTRACTOR_STATE_COLOR = 'grey'

        sts = self.generate_state_transition_system()

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

        # Draw the graph. Different layouts can be used, for a full list see
        # https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
        # 
        # A better drawing can be obtained with the PyGraphviz.AGraph class, but requires the installation of
        # PyGraphviz (https://pygraphviz.github.io/)
        nx.draw_networkx(sts,
                         with_labels=True,
                         pos=nx.spring_layout(sts),
                         node_color = node_colors,
                         font_size=8)

        plt.show()

if __name__ == "__main__":
    bn = BN(num_nodes=3, mode="synchronous")
    bn.draw_state_transition_system()