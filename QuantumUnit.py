import numpy as np
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix
from itertools import combinations

class QuantumUnit:

    def __init__(self):
        self.backend = Aer.get_backend('aer_simulator_statevector')

    @staticmethod
    def classical_to_statevector(classical_vector):
        num_qubits = len(classical_vector)
        num_states = 2 ** num_qubits
        state_index = int("".join(map(str, classical_vector)), 2)
        statevector = np.zeros(num_states, dtype=complex)
        statevector[state_index] = 1.0 + 0.0j

        return statevector

    def run_circuit(self, circuit, initial_state=None):
        """
        Execute the quantum circuit and return the final state vector.
        If an initial state vector is provided, the circuit is executed from this state.
        """
        if initial_state is not None:
            qubit_indexes = [qubit.index for qubit in circuit.qubits]
            #initial_statevector = self.classical_to_statevector(initial_state)
            init_cicuit = QuantumCircuit(len(qubit_indexes))
            init_cicuit.initialize(initial_state, qubit_indexes)
            init_cicuit = init_cicuit.compose(circuit)
            circuit = init_cicuit
        circuit.save_statevector()
        transpiled_circuit = transpile(circuit, self.backend)
        job = self.backend.run(transpiled_circuit)
        result = job.result()
        return result.get_statevector(circuit)

    def slice_circuit(self, circuit, start_layer, end_layer, start_qubit, end_qubit):
        """
        Take a quantum circuit and return a new quantum circuit that only includes
        the layers from start_layer to end_layer (inclusive) and only the qubits from
        start_qubit to end_qubit (inclusive).

        circuit: QuantumCircuit, the original quantum circuit.
        start_layer: int, the index of the starting layer.
        end_layer: int, the index of the ending layer.
        start_qubit: int, the index of the first qubit to include.
        end_qubit: int, the index of the last qubit to include.
        """
        sliced_circuit = QuantumCircuit(circuit.num_qubits)
        for layer, (gate, qubits, _) in enumerate(circuit.data):
            if start_layer <= layer <= end_layer and all(
                    start_qubit <= qubit.index <= end_qubit for qubit in qubits):
                sliced_circuit.append(gate, qubits)
        return sliced_circuit

    def is_classical_state(self, probabilities):
        """
        Check if the state described by 'probabilities' is a classical state, i.e., one of the states
        has probability 1 and the rest have probability 0.

        probabilities: np.array, a probability distribution representing the state.

        Returns: bool, True if it's a classical state, False otherwise.
        """
        real_coefficients = np.real(probabilities)
        num_ones = np.count_nonzero(real_coefficients)
        return num_ones == 1

    def assertClassicalEqual(self, circuit, initial_state, expected_state):
        """
        Check if the output state of the circuit, when run with 'initial_state', is a classical state equal
        to 'expected_state'.

        circuit: QuantumCircuit, the quantum circuit to test.
        initial_state: list of int, the initial state to run the circuit with.
        expected_state: list of int, the expected output state.

        Returns: bool, True if the output state equals the expected state, False otherwise.
        """
        # Transform the qubit array to an statevector
        initial_state = self.classical_to_statevector(initial_state)
        # Run the circuit
        probabilities = self.run_circuit(circuit, initial_state)
        # Check if the result is a classical state
        if not self.is_classical_state(probabilities):
            raise AssertionError("The output is not a classical state.")
        # Check if the state is equal to the unexpected state
        real_coefficients = np.real(probabilities)
        return np.array_equal(real_coefficients, expected_state)

    def assertClassicalNotEqual(self, circuit, initial_state, unexpected_state):
        """
        Check if the output state of the circuit, when run with 'initial_state', is a classical state not equal
        to 'unexpected_state'.

        circuit: QuantumCircuit, the quantum circuit to test.
        initial_state: list of int, the initial state to run the circuit with.
        unexpected_state: list of int, the unexpected output state.

        Returns: bool, True if the output state does not equal the unexpected state, False otherwise.
        """
        # Transform the qubit array to an statevector
        initial_statevector = self.classical_to_statevector(initial_state)
        # Run the circuit
        probabilities = self.run_circuit(circuit, initial_statevector)
        # Check if the result is a classical state
        if not self.is_classical_state(probabilities):
            raise AssertionError("The output is not a classical state.")
        # Check if the state is not equal to the unexpected state
        real_coefficients = np.real(probabilities)
        """
        print('initial_statevector:', initial_statevector)
        print('expected_state:', unexpected_state)
        print('obtained_state:',real_coefficients)
        print('not_equal?:',not (np.array_equal(real_coefficients, unexpected_state)))
        print(' ')
        """
        return not (np.array_equal(real_coefficients, unexpected_state))

    def assertClassicalLessThan(self, circuit, initial_state, greater_state):
        """
        Check if the output state of the circuit, when run with 'initial_state', is a classical state
        that represents a number less than the one represented by 'expected_state'.

        circuit: QuantumCircuit, the quantum circuit to test.
        initial_state: list of int, the initial state to run the circuit with.
        expected_state: list of int, the expected output state.

        Returns: bool, True if the output state represents a number less than the expected state, False otherwise.
        """
        # Transform the qubit array to an statevector
        initial_statevector = self.classical_to_statevector(initial_state)
        # Run the circuit
        probabilities = self.run_circuit(circuit, initial_statevector)
        # Check if the result is a classical state
        if not self.is_classical_state(probabilities):
            raise AssertionError("The output is not a classical state.")
        # Check if the state is less than the expected state
        real_coefficients = np.real(probabilities)
        # Encuentra el índice del coeficiente máximo
        max_index = np.argmax(real_coefficients)
        # Convierte el índice en una representación binaria
        binary_repr = format(max_index, 'b')
        # Rellena con ceros a la izquierda para obtener la representación completa de los qubits
        num_qubits = int(np.log2(len(real_coefficients)))
        qubit_repr = [int(bit) for bit in binary_repr.zfill(num_qubits)]
        binary_str = ''.join(map(str, qubit_repr))
        integer_lower = int(binary_str, 2)
        binary_str = ''.join(map(str, greater_state))
        integer_greater = int(binary_str, 2)
        return integer_lower < integer_greater

    def assertClassicalGreaterThan(self, circuit, initial_state, lower_state):
        """
        Check if the output state of the circuit, when run with 'initial_state', is a classical state
        that represents a number greater than the one represented by 'expected_state'.

        circuit: QuantumCircuit, the quantum circuit to test.
        initial_state: list of int, the initial state to run the circuit with.
        expected_state: list of int, the expected output state.

        Returns: bool, True if the output state represents a number greater than the expected state, False otherwise.
        """
        # Transform the qubit array to an statevector
        initial_statevector = self.classical_to_statevector(initial_state)
        # Run the circuit
        probabilities = self.run_circuit(circuit, initial_statevector)
        # Check if the result is a classical state
        if not self.is_classical_state(probabilities):
            raise AssertionError("The output is not a classical state.")
        # Check if the state is less than the expected state
        real_coefficients = np.real(probabilities)
        # Encuentra el índice del coeficiente máximo
        max_index = np.argmax(real_coefficients)
        # Convierte el índice en una representación binaria
        binary_repr = format(max_index, 'b')
        # Rellena con ceros a la izquierda para obtener la representación completa de los qubits
        num_qubits = int(np.log2(len(real_coefficients)))
        qubit_repr = [int(bit) for bit in binary_repr.zfill(num_qubits)]
        # Cast to decimal and compare
        binary_str = ''.join(map(str, qubit_repr))
        integer_greater = int(binary_str, 2)
        binary_str = ''.join(map(str, lower_state))
        integer_lower = int(binary_str, 2)
        return integer_lower < integer_greater

    def assertClassicalBetween(self, circuit, initial_state, lower_state, greater_state):
        """
        Check if the output state of the circuit, when run with 'initial_state', is a classical state
        that represents a number between the ones represented by 'lower_bound' and 'upper_bound' (inclusive).

        circuit: QuantumCircuit, the quantum circuit to test.
        initial_state: list of int, the initial state to run the circuit with.
        lower_bound: list of int, the lower bound of the output state.
        upper_bound: list of int, the upper bound of the output state.

        Returns: bool, True if the output state represents a number between the bounds, False otherwise.
        """
        # Transform the qubit array to an statevector
        initial_statevector = self.classical_to_statevector(initial_state)
        # Run the circuit
        probabilities = self.run_circuit(circuit, initial_statevector)
        # Check if the result is a classical state
        if not self.is_classical_state(probabilities):
            raise AssertionError("The output is not a classical state.")
        # Check if the state is less than the expected state
        real_coefficients = np.real(probabilities)
        # Encuentra el índice del coeficiente máximo
        max_index = np.argmax(real_coefficients)
        # Convierte el índice en una representación binaria
        binary_repr = format(max_index, 'b')
        # Rellena con ceros a la izquierda para obtener la representación completa de los qubits
        num_qubits = int(np.log2(len(real_coefficients)))
        qubit_repr = [int(bit) for bit in binary_repr.zfill(num_qubits)]
        # Cast to decimal and compare
        binary_str = ''.join(map(str, qubit_repr))
        integer_middle = int(binary_str, 2)
        binary_str = ''.join(map(str, lower_state))
        integer_lower = int(binary_str, 2)
        binary_str = ''.join(map(str, greater_state))
        integer_greater = int(binary_str, 2)
        return ((integer_lower < integer_middle) and (integer_middle < integer_greater))

    def assertStatevectorEqual(self, circuit, expected_state, initial_state=None, start_layer=None, end_layer=None, start_qubit=None, end_qubit=None):
        """
        Check if the output of the quantum circuit matches the expected state vector.
        If start and end layer/qubit are provided, the circuit is sliced before executing it.
        An initial state vector for the circuit can be provided.
        """
        if start_layer is not None and end_layer is not None and start_qubit is not None and end_qubit is not None:
            circuit = self.slice_circuit(circuit, start_layer, end_layer, start_qubit, end_qubit)
        final_state = self.run_circuit(circuit, initial_state)
        return np.isclose(final_state, expected_state).all()

    def assertDensityMatrixEqual(self, circuit, expected_density_matrix, initial_state=None, start_layer=None,
                                 end_layer=None, start_qubit=None, end_qubit=None):
        """
        Check if the output of the quantum circuit matches the expected density matrix.
        If start and end layer/qubit are provided, the circuit is sliced before executing it.
        An initial state vector for the circuit can be provided.
        """
        if start_layer is not None and end_layer is not None and start_qubit is not None and end_qubit is not None:
            circuit = self.slice_circuit(circuit, start_layer, end_layer, start_qubit, end_qubit)
        final_state = self.run_circuit(circuit, initial_state)
        final_density_matrix = DensityMatrix(final_state)
        return np.isclose(final_density_matrix.data, expected_density_matrix.data).all()

    def assertEntanglement(self, circuit, qubit1, qubit2, initial_state=None, start_layer=None, end_layer=None, start_qubit=None, end_qubit=None):
        """
        Check if two specific qubits are entangled in the output of the quantum circuit.
        If start and end layer/qubit are provided, the circuit is sliced before executing it.
        An initial state vector for the circuit can be provided.
        """
        if start_layer is not None and end_layer is not None and start_qubit is not None and end_qubit is not None:
            circuit = self.slice_circuit(circuit, start_layer, end_layer, start_qubit, end_qubit)
        final_state = self.run_circuit(circuit, initial_state)
        statevector = Statevector(final_state)
        reduced_state = partial_trace(statevector, list(set(range(circuit.num_qubits)) - set([qubit1, qubit2])))
        eigenvalues = np.linalg.eigvalsh(reduced_state.data)
        return not np.isclose(eigenvalues[1:], 0).all()

    def assertGeneralEntanglement(self, circuit, qubits, initial_state=None, start_layer=None, end_layer=None,
                                  start_qubit=None, end_qubit=None):
        """
        Check if a set of qubits are mutually entangled in the output of the quantum circuit.
        If start and end layer/qubit are provided, the circuit is sliced before executing it.
        An initial state vector for the circuit can be provided.
        """
        if start_layer is not None and end_layer is not None and start_qubit is not None and end_qubit is not None:
            circuit = self.slice_circuit(circuit, start_layer, end_layer, start_qubit, end_qubit)
        final_state = self.run_circuit(circuit, initial_state)
        statevector = Statevector(final_state)
        for subset in combinations(qubits, 2):
            reduced_state = partial_trace(statevector, list(set(range(circuit.num_qubits)) - set(subset)))
            eigenvalues = np.linalg.eigvalsh(reduced_state.data)
            if not np.isclose(eigenvalues[1:], 0).all():
                return True
        return False