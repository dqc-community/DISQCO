import matplotlib.pyplot as plt
from qiskit import transpile
import numpy as np

def check_no_cross_partition_gates(circuit, qpu_graph):
    """
    Checks that there are no invalid two-qubit gates:
    - Gates between Qi and Qj (i ≠ j) are NOT allowed.
    - Gates between Ci and Qi are allowed.
    - Gates between Ci and Cj are allowed ONLY if i and j are connected in qpu_graph.
    Returns True if valid, False otherwise. Prints offending gates if found.
    """
    # Map qubits to their register index and type (Q for data, C for comm)
    qubit_to_reg = {}
    qubit_to_type = {}
    for reg in circuit.qregs:
        if reg.name.startswith("Q"):
            reg_idx = int(reg.name[1:].split("_")[0])
            for q in reg:
                qubit_to_reg[q] = reg_idx
                qubit_to_type[q] = "Q"
        elif reg.name.startswith("C"):
            reg_idx = int(reg.name[1:].split("_")[0])
            for q in reg:
                qubit_to_reg[q] = reg_idx
                qubit_to_type[q] = "C"
    valid = True
    for instr, qargs, _ in circuit.data:
        if instr.num_qubits == 2:
            types = [qubit_to_type.get(q, None) for q in qargs]
            regs = [qubit_to_reg.get(q, None) for q in qargs]
            # Check for Qi-Qj (i ≠ j)
            if types[0] == "Q" and types[1] == "Q" and regs[0] != regs[1]:
                print(f"Invalid gate {instr.name} between data registers Q{regs[0]} and Q{regs[1]} on qubits {qargs}")
                valid = False
            # Check for Ci-Qj or Qi-Cj (allowed)
            elif (types[0] == "C" and types[1] == "Q") or (types[0] == "Q" and types[1] == "C"):
                continue
            # Check for Ci-Cj
            elif types[0] == "C" and types[1] == "C":
                if regs[0] == regs[1]:
                    continue
                elif qpu_graph.has_edge(regs[0], regs[1]) or qpu_graph.has_edge(regs[1], regs[0]):
                    continue
                else:
                    print(f"Invalid gate {instr.name} between comm registers C{regs[0]} and C{regs[1]} (not connected) on qubits {qargs}")
                    valid = False
    return valid

def run_sampler(circuit, shots=4096):
    from qiskit_aer.primitives import SamplerV2
    sampler = SamplerV2()
    num_qubits = circuit.num_qubits
    dec_circuit = circuit.copy()
    dec_circuit = transpile(dec_circuit, optimization_level=0)
    dec_circuit = dec_circuit.decompose()
    if num_qubits <= 13:

        job = sampler.run([dec_circuit], shots=shots)
        job_result = job.result()
        data = job_result[0].data
    else:
        print("Too many qubits")
        data = None
    return data

def plot(data, labels=False):
    from qiskit.visualization import plot_histogram
    if data is None:
        print("No data to plot")
        return
    if 'result' in data:
        info = data['result']
    elif 'meas' in data:
        info = data['meas']
    elif 'measure' in data:
        info = data['measure']
    else:
        print("No data to plot")
        return

    counts_base = info.get_counts()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_histogram(counts_base, bar_labels=False, ax=ax)
    if not labels:
        ax.set_xticks([])

def get_fidelity(data1, data2, shots):
    if data1 is None or data2 is None:
        print("No data to compare")
        return None
    if 'result' in data1:
        info1 = data1['result']
    else:
        info1 = data1['meas']

    if 'result' in data2:
        info2 = data2['result']
    else:
        info2 = data2['meas']
    
    counts1 = info1.get_counts()
    counts2 = info2.get_counts()
    for key in counts1:
        digits = len(key)
        break
    norm = 0    
    max_string = '1'*digits
    integer = int(max_string, 2)
    for i in range(integer+1):
        binary = bin(i)
        binary = binary[2:]
        binary = '0'*(digits-len(binary)) + binary
        if binary in counts1:
            counts1_val = counts1[binary]/shots
        else:
            counts1_val = 0
        if binary in counts2:
            counts2_val = counts2[binary]/shots
        else:
            counts2_val = 0
        norm += np.abs(counts1_val - counts2_val)
    return 1 - norm**2
