from __future__ import annotations

import argparse
import pathlib

import numpy as np
from disqco import QuantumCircuitHyperGraph, QuantumNetwork
from disqco.graphs.hypergraph_methods import calculate_full_cost
from disqco.parti.partitioner import QuantumCircuitPartitioner
from qiskit import QuantumCircuit, qasm2, transpile

CIRCUITS_DIR = pathlib.Path(__file__).parent / "circuits"

# (qubits_per_node, nodes)
CASES: dict[str, tuple[int, int]] = {
    "qft_n18":          (9,  2),
    # "square_root_n18":  (9,  2),
    "adder_n28":        (14, 2),
    "ising_n34":        (17, 2),
    "ghz_n40":          (20, 2),
    "qft_n63":          (32, 2),
    "adder_n64":        (32, 2),
    "wstate_n76":       (38, 2),
    "wstate_n118":      (59, 2),
    "adder_n118":       (59, 2),
}


def _normalize(qiskit_circuit: QuantumCircuit) -> QuantumCircuit:
    """Restrict to unitary 1q/2q ops and decompose unsupported gates."""
    stripped = QuantumCircuit(qiskit_circuit.num_qubits)
    for ci in qiskit_circuit.data:
        name = ci.operation.name.lower()
        if name in {"measure", "reset", "barrier", "if_else", "delay"}:
            continue
        if ci.clbits:
            continue
        remapped_qubits = [stripped.qubits[qiskit_circuit.find_bit(q).index] for q in ci.qubits]
        stripped.append(ci.operation, remapped_qubits, [])

    return transpile(stripped, basis_gates=["u", "cp", "cx"], optimization_level=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-passes", type=int, default=20)
    parser.add_argument("--partitioner", choices=["fm", "fgp", "genetic"], default="fm")
    args = parser.parse_args()

    rows: list[tuple[str, int, int, int]] = []

    for circuit_id, (qpn, n_nodes) in CASES.items():
        qasm_text = (CIRCUITS_DIR / f"{circuit_id}.qasm").read_text()
        qc = _normalize(qasm2.loads(qasm_text))

        net = QuantumNetwork.create([qpn] * n_nodes, "all_to_all")
        part = QuantumCircuitPartitioner.create(args.partitioner, qc, net)
        res = part.partition(num_passes=args.num_passes, log=False)
        assignment = np.asarray(res["best_assignment"], dtype=int)

        hg = QuantumCircuitHyperGraph(qc)
        hyperedge_cost = calculate_full_cost(hg, assignment, n_nodes)

        rows.append((circuit_id, n_nodes, qpn, hyperedge_cost))
        _print_table(rows)


def _print_table(rows: list[tuple[str, int, int, int]]) -> None:
    col_w = max(len(r[0]) for r in rows)
    print(f"\n{'circuit':<{col_w}}  nodes  qpn  hyperedge_cost")
    print("-" * (col_w + 30))
    for name, n_nodes, qpn, hc in rows:
        print(f"{name:<{col_w}}  {n_nodes:<5}  {qpn:<3}  {hc}")


if __name__ == "__main__":
    main()
