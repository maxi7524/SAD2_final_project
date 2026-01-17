import unittest
from boolean import BooleanAlgebra
from bn import BN  # replace with the actual filename of your BN script

class TestBN(unittest.TestCase):
    def setUp(self):
        algebra = BooleanAlgebra()
        self.x1 = algebra.Symbol('x0')
        self.x2 = algebra.Symbol('x1')
        self.x3 = algebra.Symbol('x2')

        # Define functions
        f1 = self.x2
        f2 = ~self.x2
        f3 = ~self.x2 | self.x3

        self.functions = [f1, f2, f3]

        # Create BN with fixed functions
        self.bn = BN(num_nodes=3, mode="synchronous", sampling_frequency=1, trajectory_length=5, functions=self.functions)

    def test_synchronous_update(self):
        # Initial state (x0, x1, x2) = (0, 0, 0)
        state = (0, 0, 0)
        next_state = self.bn.next_synchronous(state)
        # f1=x2=0, f2=¬x2=1, f3=¬x2∨x3=1
        self.assertEqual(next_state, (0, 1, 1))

        state = (1, 1, 0)
        next_state = self.bn.next_synchronous(state)
        # f1=x2=1, f2=¬x2=0, f3=¬x2∨x3=0
        self.assertEqual(next_state, (1, 0, 0))

    def test_asynchronous_update(self):
        state = (0, 0, 0)
        # Update coordinate 0 (f1=x2)
        next_state = self.bn.next_asynchronous(state, 0)
        self.assertEqual(next_state, (0, 0, 0))  # x2=0

        # Update coordinate 1 (f2=¬x2)
        next_state = self.bn.next_asynchronous(state, 1)
        self.assertEqual(next_state, (0, 1, 0))

        # Update coordinate 2 (f3=¬x2 ∨ x3)
        next_state = self.bn.next_asynchronous(state, 2)
        self.assertEqual(next_state, (0, 0, 1))


if __name__ == '__main__':
    unittest.main()
