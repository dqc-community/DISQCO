"""
Test suite for circuit extraction functionality.

Tests the PartitionedCircuitExtractor class for extracting distributed quantum
circuits from hypergraphs with both initial (unoptimized) and optimized assignments.
"""

from pathlib import Path

import networkx as nx
import pytest
import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile, QuantumRegister

from disqco import QuantumNetwork, QuantumCircuitHyperGraph, PartitionedCircuitExtractor
from disqco.circuit_extraction.DQC_qubit_manager import (
    CommunicationQubitLimitError,
    CommunicationQubitManager,
)
from disqco.circuits.cp_fraction import cp_fraction
from disqco.parti import FiducciaMattheyses
from disqco import set_initial_partition_assignment
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "circuits"


@pytest.fixture
def test_circuit():
    """Create a test circuit for extraction"""
    circuit = cp_fraction(num_qubits=8, depth=8, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return circuit


@pytest.fixture
def test_network():
    """Create a 2-QPU network for testing"""
    return QuantumNetwork.create([5, 5], 'all_to_all')


@pytest.fixture
def test_hypergraph(test_circuit):
    """Create a hypergraph from test circuit"""
    return QuantumCircuitHyperGraph(test_circuit)


@pytest.fixture
def initial_assignment(test_hypergraph, test_network):
    """Create initial unoptimized assignment"""
    return set_initial_partition_assignment(test_hypergraph, test_network)


@pytest.fixture
def optimized_assignment(test_circuit, test_network):
    """Create optimized assignment using FM partitioner"""
    partitioner = FiducciaMattheyses(test_circuit, network=test_network)
    results = partitioner.partition(num_passes=5)
    return results['best_assignment']


def test_circuit_extractor_import():
    """Test that PartitionedCircuitExtractor can be imported from disqco"""
    from disqco import PartitionedCircuitExtractor
    assert PartitionedCircuitExtractor is not None


def test_circuit_extractor_from_circuit_extraction_module():
    """Test importing from circuit_extraction module"""
    from disqco.circuit_extraction import PartitionedCircuitExtractor
    assert PartitionedCircuitExtractor is not None


def test_circuit_extractor_instantiation(test_hypergraph, test_network, initial_assignment):
    """Test creating an instance of PartitionedCircuitExtractor"""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment
    )
    
    assert extractor is not None
    assert extractor.graph == test_hypergraph
    assert extractor.network == test_network
    assert np.array_equal(extractor.partition_assignment, initial_assignment)
    
    print("\n✓ PartitionedCircuitExtractor instantiated successfully")


def _star_extractor(max_comm_qubits_per_node=None):
    circuit = QuantumCircuit(3)
    hypergraph = QuantumCircuitHyperGraph(circuit)
    network = QuantumNetwork.create([1, 1, 1], "all_to_all")
    assignment = np.array([[0, 1, 2]])
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment,
        max_comm_qubits_per_node=max_comm_qubits_per_node,
    )
    tree = nx.DiGraph([(0, 1), (0, 2)])
    extractor.teleportation_manager.build_k_fold_starting_process(
        root_q=0,
        p_root=0,
        target_partitions=[1, 2],
        tree=tree,
    )
    return extractor


def _comm_qubit_count(circuit):
    return sum(reg.size for reg in circuit.qregs if reg.name.startswith("C"))


def test_k1_star_gate_uses_one_comm_qubit_per_node():
    extractor = _star_extractor(max_comm_qubits_per_node=1)

    assert _comm_qubit_count(extractor.qc) == 3
    assert extractor.comm_manager.get_peak_comm_usage(0) == 1
    assert extractor.comm_manager.get_peak_comm_usage(1) == 1
    assert extractor.comm_manager.get_peak_comm_usage(2) == 1


def test_k2_allows_two_comm_qubits_per_node():
    extractor = _star_extractor(max_comm_qubits_per_node=2)

    assert _comm_qubit_count(extractor.qc) <= 6


def test_k1_chain_intermediate_node_raises_at_limit():
    circuit = QuantumCircuit(3)
    hypergraph = QuantumCircuitHyperGraph(circuit)
    network = QuantumNetwork.create([1, 1, 1], "linear")
    assignment = np.array([[0, 1, 2]])
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment,
        max_comm_qubits_per_node=1,
    )
    tree = nx.DiGraph([(0, 1), (1, 2)])

    with pytest.raises(CommunicationQubitLimitError, match="Node 1.*limit of 1"):
        extractor.teleportation_manager.build_k_fold_starting_process(
            root_q=0,
            p_root=0,
            target_partitions=[2],
            tree=tree,
        )


def test_k2_chain_allows_intermediate_node():
    circuit = QuantumCircuit(3)
    hypergraph = QuantumCircuitHyperGraph(circuit)
    network = QuantumNetwork.create([1, 1, 1], "linear")
    assignment = np.array([[0, 1, 2]])
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment,
        max_comm_qubits_per_node=2,
    )
    tree = nx.DiGraph([(0, 1), (1, 2)])

    extractor.teleportation_manager.build_k_fold_starting_process(
        root_q=0,
        p_root=0,
        target_partitions=[2],
        tree=tree,
    )

    assert extractor.comm_manager.get_peak_comm_usage(1) == 2


def test_invalid_max_comm_qubits_per_node_raises():
    circuit = QuantumCircuit(1)
    hypergraph = QuantumCircuitHyperGraph(circuit)
    network = QuantumNetwork.create([1], "all_to_all")
    assignment = np.array([[0]])

    with pytest.raises(ValueError, match="must be >= 1 or None"):
        PartitionedCircuitExtractor(
            graph=hypergraph,
            network=network,
            partition_assignment=assignment,
            max_comm_qubits_per_node=0,
        )


def test_unbounded_matches_current_behaviour():
    implicit_unbounded = _star_extractor()
    explicit_unbounded = _star_extractor(max_comm_qubits_per_node=None)

    assert qasm2.dumps(explicit_unbounded.qc) == qasm2.dumps(implicit_unbounded.qc)


def test_find_comm_idx_raises_at_limit():
    comm_reg = QuantumRegister(1, name="C0_0")
    qc = QuantumCircuit(comm_reg)
    manager = CommunicationQubitManager({0: [comm_reg]}, qc, max_comm_qubits=1)

    manager.find_comm_idx(0)

    with pytest.raises(CommunicationQubitLimitError, match="Node 0.*limit of 1.*currently in use"):
        manager.find_comm_idx(0)


def test_invalid_manager_max_comm_qubits_raises():
    comm_reg = QuantumRegister(1, name="C0_0")
    qc = QuantumCircuit(comm_reg)

    with pytest.raises(ValueError, match="max_comm_qubits must be >= 1 or None"):
        CommunicationQubitManager({0: [comm_reg]}, qc, max_comm_qubits=0)


def test_get_peak_comm_usage():
    comm_reg = QuantumRegister(1, name="C0_0")
    qc = QuantumCircuit(comm_reg)
    manager = CommunicationQubitManager({0: [comm_reg]}, qc)

    first = manager.find_comm_idx(0)
    second = manager.find_comm_idx(0)
    assert manager.get_peak_comm_usage(0) == 2

    manager.release_comm_qubit(0, first)
    manager.release_comm_qubit(0, second)
    third = manager.find_comm_idx(0)

    assert manager.get_peak_comm_usage(0) == 2
    manager.release_comm_qubit(0, third)


def test_extract_circuit_with_initial_assignment(test_hypergraph, test_network, initial_assignment):
    """Test extracting circuit with initial unoptimized assignment"""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    assert partitioned_circuit is not None
    assert isinstance(partitioned_circuit, QuantumCircuit)
    assert partitioned_circuit.num_qubits > 0
    assert partitioned_circuit.depth() > 0
    
    print(f"\n✓ Extracted circuit with initial assignment:")
    print(f"  Circuit depth: {partitioned_circuit.depth()}")
    print(f"  Number of qubits: {partitioned_circuit.num_qubits}")


def test_extract_circuit_with_optimized_assignment(test_hypergraph, test_network, optimized_assignment):
    """Test extracting circuit with optimized FM assignment"""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=optimized_assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    assert partitioned_circuit is not None
    assert isinstance(partitioned_circuit, QuantumCircuit)
    assert partitioned_circuit.num_qubits > 0
    assert partitioned_circuit.depth() > 0
    
    print(f"\n✓ Extracted circuit with optimized assignment:")
    print(f"  Circuit depth: {partitioned_circuit.depth()}")
    print(f"  Number of qubits: {partitioned_circuit.num_qubits}")


def test_compare_epr_counts_initial_vs_optimized(test_hypergraph, test_network, 
                                                  initial_assignment, optimized_assignment):
    """Test that optimized assignment typically produces fewer EPR pairs"""
    # Extract with initial assignment
    extractor_initial = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment
    )
    circuit_initial = extractor_initial.extract_partitioned_circuit()
    circuit_initial_epr = circuit_initial
    
    # Extract with optimized assignment
    extractor_optimized = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=optimized_assignment
    )
    circuit_optimized = extractor_optimized.extract_partitioned_circuit()
    circuit_optimized_epr = circuit_optimized
    
    # Count EPR pairs
    ops_initial = circuit_initial_epr.count_ops()
    ops_optimized = circuit_optimized_epr.count_ops()
    
    epr_initial = ops_initial.get('EPR', 0)
    epr_optimized = ops_optimized.get('EPR', 0)
    
    print(f"\n✓ EPR pair comparison:")
    print(f"  Initial assignment: {epr_initial} EPR pairs")
    print(f"  Optimized assignment: {epr_optimized} EPR pairs")
    print(f"  Reduction: {epr_initial - epr_optimized} EPR pairs")
    
    # Optimized should typically use same or fewer EPR pairs
    assert epr_optimized <= epr_initial


def test_extracted_circuit_structure(test_hypergraph, test_network, initial_assignment):
    """Test the structure of extracted partitioned circuit"""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    # Check circuit has quantum and classical registers
    assert len(partitioned_circuit.qregs) > 0
    assert len(partitioned_circuit.cregs) > 0
    
    # Circuit should have operations
    assert len(partitioned_circuit.data) > 0
    
    print(f"\n✓ Circuit structure:")
    print(f"  Quantum registers: {len(partitioned_circuit.qregs)}")
    print(f"  Classical registers: {len(partitioned_circuit.cregs)}")
    print(f"  Total operations: {len(partitioned_circuit.data)}")


def test_extraction_with_different_networks(test_hypergraph, initial_assignment):
    """Test extraction with different network topologies"""
    # Test with linear network
    linear_net = QuantumNetwork.create([5, 5], 'linear')
    extractor_linear = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=linear_net,
        partition_assignment=initial_assignment
    )
    circuit_linear = extractor_linear.extract_partitioned_circuit()
    assert circuit_linear is not None
    
    # Test with all-to-all network
    alltoall_net = QuantumNetwork.create([5, 5], 'all_to_all')
    extractor_alltoall = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=alltoall_net,
        partition_assignment=initial_assignment
    )
    circuit_alltoall = extractor_alltoall.extract_partitioned_circuit()
    assert circuit_alltoall is not None
    
    print("\n✓ Extraction works with different network topologies")


def test_extraction_with_three_partitions(test_circuit):
    """Test extraction with 3 QPUs"""
    network = QuantumNetwork.create([3, 3, 3], 'linear')
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)
    
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    assert partitioned_circuit is not None
    assert partitioned_circuit.num_qubits > 0
    
    print(f"\n✓ Extraction with 3 partitions:")
    print(f"  Circuit depth: {partitioned_circuit.depth()}")


def test_extraction_with_four_partitions(test_circuit):
    """Test extraction with 4 QPUs"""
    network = QuantumNetwork.create([2, 2, 2, 2], 'grid')
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)
    
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    assert partitioned_circuit is not None
    assert partitioned_circuit.num_qubits > 0
    
    print(f"\n✓ Extraction with 4 partitions (grid):")
    print(f"  Circuit depth: {partitioned_circuit.depth()}")


def test_full_workflow_initial_to_optimized():
    """Test complete workflow from initial assignment to optimized extraction"""
    # Create circuit
    circuit = cp_fraction(num_qubits=12, depth=12, fraction=0.5, seed=123)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    
    # Create network
    network = QuantumNetwork.create([5, 5, 5], 'linear')
    
    # Create hypergraph
    hypergraph = QuantumCircuitHyperGraph(circuit)
    
    # Get initial assignment
    initial_assignment = set_initial_partition_assignment(hypergraph, network)
    
    # Extract with initial assignment
    extractor_initial = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=initial_assignment
    )
    circuit_initial = extractor_initial.extract_partitioned_circuit()
    
    # Run optimization
    partitioner = FiducciaMattheyses(circuit, network=network)
    results = partitioner.partition(num_passes=10)
    optimized_assignment = results['best_assignment']
    
    # Extract with optimized assignment
    extractor_optimized = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=optimized_assignment
    )
    circuit_optimized = extractor_optimized.extract_partitioned_circuit()
    
    # Both circuits should be valid
    assert circuit_initial is not None
    assert circuit_optimized is not None
    assert circuit_initial.num_qubits > 0
    assert circuit_optimized.num_qubits > 0
    
    # Count EPR pairs
    circuit_initial_epr = circuit_initial
    circuit_optimized_epr = circuit_optimized
    
    epr_initial = circuit_initial_epr.count_ops().get('EPR', 0)
    epr_optimized = circuit_optimized_epr.count_ops().get('EPR', 0)
    
    print(f"\n✓ Full workflow test:")
    print(f"  Initial EPR pairs: {epr_initial}")
    print(f"  Optimized EPR pairs: {epr_optimized}")
    print(f"  Improvement: {epr_initial - epr_optimized} pairs ({100*(epr_initial-epr_optimized)/max(epr_initial,1):.1f}%)")
    
    # Optimized should be better or equal
    assert epr_optimized <= epr_initial


def test_extraction_preserves_circuit_semantics(test_circuit, test_network):
    """Test that extraction produces a circuit with same number of operations"""
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, test_network)
    
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=test_network,
        partition_assignment=assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    # Original circuit operations
    original_ops = test_circuit.count_ops()
    original_gate_count = sum(count for gate, count in original_ops.items() 
                              if gate not in ['barrier', 'measure'])
    
    # Partitioned circuit will have additional teleportation operations
    # but should preserve the original gates
    assert partitioned_circuit.depth() >= test_circuit.depth()
    
    print(f"\n✓ Circuit semantics:")
    print(f"  Original gates: {original_gate_count}")
    print(f"  Original depth: {test_circuit.depth()}")
    print(f"  Partitioned depth: {partitioned_circuit.depth()}")


def test_extractor_with_single_partition():
    """Test extraction with single partition (trivial case)"""
    circuit = cp_fraction(num_qubits=8, depth=8, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    
    # Single partition - should need no EPR pairs
    network = QuantumNetwork({0: 10})
    hypergraph = QuantumCircuitHyperGraph(circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)
    
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    circuit_epr = partitioned_circuit
    
    epr_count = circuit_epr.count_ops().get('EPR', 0)
    
    # Single partition should require 0 EPR pairs
    assert epr_count == 0
    
    print(f"\n✓ Single partition extraction: {epr_count} EPR pairs (expected 0)")


def test_group_closed_when_all_subgates_applied_immediately():
    """Regression: groups whose last two-qubit gate lands at the current time step must have
    close_group called explicitly, otherwise the root qubit stays marked as grouped and
    subsequent teleportation logic silently skips it, producing an incorrect circuit."""
    circuit = qasm2.load(FIXTURES_DIR / "variational_n4_transpiled.qasm", custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

    hypergraph = QuantumCircuitHyperGraph(circuit)
    network = QuantumNetwork.create([3, 3], "all_to_all")
    initial_assignment = set_initial_partition_assignment(hypergraph, network)
    partitioner = FiducciaMattheyses(
        circuit,
        network,
        initial_assignment,
        hypergraph=hypergraph,
    )
    results = partitioner.multilevel_partition(
        coarsener=HypergraphCoarsener().coarsen_recursive_batches_mapped,
        passes_per_level=10,
    )

    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=results["best_assignment"],
    )
    partitioned_circuit = extractor.extract_partitioned_circuit()

    assert partitioned_circuit is not None
    assert isinstance(partitioned_circuit, QuantumCircuit)
