import unittest
import numpy as np
from qiskit import QuantumCircuit
from QuantumUnit import QuantumUnit
from qiskit.quantum_info import DensityMatrix



class TestQuantumUnit(unittest.TestCase):

    def setUp(self):
        self.quantum_unit = QuantumUnit()

    def test_run_circuit(self):
        """
        Test the run_circuit method of the QuantumUnit class.
        """
        # Create a simple quantum circuit
        qc = QuantumCircuit(1)
        qc.x(0)  # Apply X gate to qubit 0, which flips it from |0> to |1>

        # The expected final state is |1> represented as [0, 1]
        expected_state = np.array([0, 1])
        final_state = self.quantum_unit.run_circuit(qc)
        self.assertTrue(np.allclose(final_state, expected_state), "The final state does not match the expected state.")
        print('test_run_circuit succeeded!')

    def test_is_classical_state(self):
        """
        Test the is_classical_state method of the QuantumUnit class.
        """
        # A classical state (e.g., |0>) has probability 1 for one state and 0 for all others
        classical_state_probabilities = np.array([1, 0])
        non_classical_state_probabilities = np.array([0.5, 0.5])

        self.assertTrue(self.quantum_unit.is_classical_state(classical_state_probabilities),
                        "Expected to return True for a classical state.")
        self.assertFalse(self.quantum_unit.is_classical_state(non_classical_state_probabilities),
                         "Expected to return False for a non-classical state.")
        print('test_is_classical_state succeeded!')

    def test_assertClassicalEqual_true(self):
        """
        Test for the assertClassicalEqual method.
        """
        test_circuit = QuantumCircuit(2)
        initial_state = [0, 0]  #[qn, ... q1, q0]
        expected_state = [0, 1, 0, 0] #[prob0, prob1, ...]
        test_circuit.x(0)
        self.assertTrue(
            self.quantum_unit.assertClassicalEqual(test_circuit, initial_state, expected_state),
            "test_assertClassicalEqual_true failed when it should have succeeded."
        )
        print('test_assertClassicalEqual_true succeeded!')

    def test_assertClassicalEqual_false(self):
        """
        Test for the assertClassicalEqual method.
        """
        test_circuit = QuantumCircuit(1)
        initial_state = [1] # Remember this is an individual state array
        unexpected_state = [0, 1] # The wrong state vector
        test_circuit.x(0)
        self.assertFalse(
            self.quantum_unit.assertClassicalEqual(test_circuit, initial_state, unexpected_state),
            "test_assertClassicalEqual_false succeeded when it should have failed."
        )
        print('test_assertClassicalEqual_false succeeded!')

    def test_assertClassicalNotEqual_true(self):
        """
        Test for the assertClassicalNotEqual method.
        """
        test_circuit = QuantumCircuit(2)
        initial_state = [1, 0]          #[qn, ... q1, q0]
        unexpected_state = [0, 1, 0, 0] #[prob0, prob1, ...]
        test_circuit.x(0)
        self.assertTrue(
            self.quantum_unit.assertClassicalNotEqual(test_circuit, initial_state, unexpected_state),
            "test_assertClassicalNotEqual_true failed when it should have succeeded."
        )
        print('test_assertClassicalNotEqual_true succeeded!')

    def test_assertClassicalNotEqual_false(self):
        """
        Test for the assertClassicalNotEqual method.
        """
        test_circuit = QuantumCircuit(2)
        initial_state = [0, 1] # Remember this is an individual state array
        test_circuit.x(0)
        expected_state = [1, 0, 0, 0] # The wrong state vector
        self.assertFalse(
            self.quantum_unit.assertClassicalNotEqual(test_circuit, initial_state, expected_state),
            "test_assertClassicalNotEqual_false succeeded when it should have failed."
        )
        print('test_assertClassicalNotEqual_false succeeded!')

    def test_assertClassicalLessThan(self):
        """
        Test for the assertClassicalLessThan method.
        """
        lower_bound_state = [0, 0, 0]  # Binary for 0
        upper_bound_state = [1, 0, 0]  # Binary for 4
        between_circuit = QuantumCircuit(3)
        between_circuit.x(1)  # Set the circuit to |010> = 2 in decimal.
        self.assertTrue(
            self.quantum_unit.assertClassicalLessThan(between_circuit, lower_bound_state, upper_bound_state),
            "assertClassicalLessThan failed when it should have succeeded."
        )
        print('test_assertClassicalLessThan_true succeeded!')
        # Counterexample
        upper_bound_state = [0, 0, 0]  # Binary for 0
        self.assertFalse(
            self.quantum_unit.assertClassicalLessThan(between_circuit, lower_bound_state, upper_bound_state),
            "assertClassicalLessThan succeeded when it should have failed."
        )
        print('test_assertClassicalLessThan_false succeeded!')

    def test_assertClassicalGreaterThan(self):
        """
        Test for the assertClassicalLessThan method.
        """
        lower_bound_state = [0, 0, 0]  # Binary for 0
        upper_bound_state = [0, 1, 0]  # Binary for 2
        between_circuit = QuantumCircuit(3)
        between_circuit.x(0)
        between_circuit.x(1)  # Set the circuit to |011> = 3 in decimal.
        self.assertTrue(
            self.quantum_unit.assertClassicalGreaterThan(between_circuit, lower_bound_state, upper_bound_state),
            "assertClassicalGreaterThan failed when it should have succeeded."
        )
        print('test_assertClassicalGreaterThan_true succeeded!')
        # Counterexample
        upper_bound_state = [1, 0, 0]  # Binary for 4
        self.assertFalse(
            self.quantum_unit.assertClassicalGreaterThan(between_circuit, lower_bound_state, upper_bound_state),
            "assertClassicalGreaterThan succeeded when it should have failed."
        )
        print('test_assertClassicalGreaterThan_false succeeded!')

    def test_assertClassicalBetween(self):
        """
        Test for the assertClassicalBetween method.
        """
        initial_state = [0, 0, 0]
        lower_bound_state = [0, 0, 0]  # Binary for 0
        upper_bound_state = [1, 0, 0]  # Binary for 4
        between_circuit = QuantumCircuit(3)
        between_circuit.x(1)  # Set the circuit to |010> = 2 in decimal.
        self.assertTrue(
            self.quantum_unit.assertClassicalBetween(between_circuit, initial_state, lower_bound_state,
                                                     upper_bound_state),
            "assertClassicalBetween failed when it should have succeeded."
        )
        print('test_assertClassicalBetween_true succeeded!')

        # Counterexample
        upper_bound_state = [0, 1, 0]  # Binary for 2
        self.assertFalse(
            self.quantum_unit.assertClassicalBetween(between_circuit, initial_state, lower_bound_state,
                                                     upper_bound_state),
            "assertClassicalBetween succeeded when it should have failed."
        )
        print('test_assertClassicalBetween_false succeeded!')

    def test_assertStatevectorEqual(self):
        """
        Test for the assertStatevectorEqual method.
        """
        test_circuit = QuantumCircuit(1)
        test_circuit.h(0)  # Apply Hadamard gate to put the qubit into a superposition state.
        expected_statevector = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2])
        result = self.quantum_unit.assertStatevectorEqual(
            test_circuit,
            expected_statevector
        )
        self.assertTrue(result, "assertStatevectorEqual failed when it should have succeeded.")
        print('test_assertStatevectorEqual succeeded!')

    def test_assertEntanglement(self):
        """
        Test for the assertEntanglement method.
        """
        test_circuit = QuantumCircuit(2)
        test_circuit.h(0)  # Apply Hadamard gate to first qubit.
        test_circuit.cx(0, 1)  # Apply CNOT gate controlled by first qubit onto second qubit.
        result = self.quantum_unit.assertEntanglement(
            test_circuit,
            0,
            1,
        )
        self.assertTrue(result, "assertEntanglement failed when it should have succeeded.")
        print('test_assertEntanglement succeeded!')

    def test_assertGeneralEntanglement(self):
        """
        Test for the assertGeneralEntanglement method.
        """
        test_circuit = QuantumCircuit(3)
        test_circuit.h(0)  # Apply Hadamard gate to first qubit.
        test_circuit.cx(0, 1)  # Apply CNOT gate controlled by first qubit onto second qubit.
        test_circuit.cx(0, 2)  # Apply CNOT gate controlled by first qubit onto third qubit.
        result = self.quantum_unit.assertGeneralEntanglement(
            test_circuit,
            [0, 1, 2],
        )
        self.assertTrue(result, "assertGeneralEntanglement failed when it should have succeeded.")
        print('test_assertGeneralEntanglement succeeded!')

    def test_slice_circuit_1(self):
         # Create a quantum circuit
        test_circuit = QuantumCircuit(3)
        test_circuit.h(0)
        test_circuit.cnot(0, 1)
        test_circuit.cnot(1, 2)

        # Slice the circuit
        sliced_circuit = self.quantum_unit.slice_circuit(test_circuit, 1, 2, 1, 2)
        result = self.quantum_unit.assertEntanglement(
            sliced_circuit,
            1,
            2,
        )
        self.assertTrue(result, "test_slice_circuit failed -> entanglement test.")

        print('test_slice_circuit_1 succeeded!')

    def test_slice_circuit_2(self):
        # Create a quantum circuit
        test_circuit = QuantumCircuit(3)
        test_circuit.h(0)
        test_circuit.cnot(0, 1)
        test_circuit.cnot(1, 2)

        # Slice the circuit
        sliced_circuit = self.quantum_unit.slice_circuit(test_circuit, 1, 2, 1, 2)
        initial_state = [0, 0, 0]  # [qn, ... q1, q0]
        expected_state = [1, 0, 0, 0, 0, 0, 0, 0]  # [prob0, prob1, ...]
        self.assertTrue(
            self.quantum_unit.assertClassicalEqual(sliced_circuit, initial_state, expected_state),
            "test_slice_circuit_2 failed -> value test."
        )
        print('test_slice_circuit_2 succeeded!')

if __name__ == "__main__":
    unittest.main()