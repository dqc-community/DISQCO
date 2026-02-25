from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.transpiler.passes import RemoveBarriers

def get_reg_mapping(circuit):
    qubit_indeces = {}
    index = 0
    for reg in circuit.qregs:
        for n in range(reg.size):
            qubit_indeces[(reg.name,n)] = index
            index += 1
    return qubit_indeces

def get_clbit_mapping(circuit):
    """Return a mapping from (creg_name, bit_index) -> global classical bit index.

    This flattens all classical registers into a single contiguous index space,
    mirroring how get_reg_mapping handles qubits.
    """
    clbit_indices = {}
    index = 0
    for creg in circuit.cregs:
        for n in range(creg.size):
            clbit_indices[(creg.name, n)] = index
            index += 1
    return clbit_indices

def circuit_to_gate_layers(circuit, qpu_sizes = None):
    "Uses qiskit DAG circuit to group gates into sublists by layer/timestep of the circuit"
    # Remove barriers to avoid empty layers
    circuit = RemoveBarriers()(circuit)
    dag = circuit_to_dag(circuit)
    layers = list(dag.multigraph_layers())
    layer_gates = []
    qubit_mapping = get_reg_mapping(circuit)
    clbit_mapping = get_clbit_mapping(circuit)
    if qpu_sizes is not None:
        max_pairs = find_max_interactions(qpu_sizes)
    for layer in layers:
        pairs = 0
        layer_info = []
        for node in layer:
            if isinstance(node, DAGOpNode):
                # Print all info in DAGOpNode
                gate_name = node.name
                qubits = [qubit_mapping[(qubit._register.name,qubit._index)] for qubit in node.qargs]
                registers = [qubit._register.name for qubit in node.qargs]
                params = node.op.params

                # Collect classical bit args (e.g., measurement targets) and classical conditions
                meta = {}
                # DAGOpNode exposes classical args via `cargs`
                cargs = getattr(node, 'cargs', []) or []
                if cargs:
                    try:
                        cargs_idx = [clbit_mapping[(bit._register.name, bit._index)] for bit in cargs]
                        meta['cargs_idx'] = cargs_idx
                        meta['cargs_regs'] = [bit._register.name for bit in cargs]
                    except Exception:
                        # Best-effort fallback if internal attributes change
                        meta['cargs_idx'] = []
                        meta['cargs_regs'] = []

                # Condition may exist on DAG node or on the underlying op
                cond = getattr(node, 'condition', None)
                if cond is None:
                    cond = getattr(node.op, 'condition', None)
                if cond is not None:
                    try:
                        cond_lhs, cond_val = cond
                        # Duck-typing: if object has 'size' treat as register, otherwise treat as single bit
                        if hasattr(cond_lhs, 'size') and getattr(cond_lhs, 'size') is not None:
                            # Register-wide condition: store register name and value for reference
                            reg_name = getattr(cond_lhs, 'name', None)
                            meta['cond_register'] = reg_name
                            meta['cond_val'] = int(cond_val)
                            # Derive a single controlling bit when possible:
                            # - If the register is size 1, it's that bit
                            # - If cond_val is a power of two within register size, use that bit position
                            # try:
                            #     reg_size = int(getattr(cond_lhs, 'size'))
                            #     if reg_name is not None:
                            #         if reg_size == 1:
                            #             meta['cond_bit'] = clbit_mapping.get((reg_name, 0))
                            #         else:
                            #             val = int(cond_val)
                            #             if val > 0 and (val & (val - 1)) == 0:
                            #                 # power of two -> single bit position
                            #                 bit_pos = (val.bit_length() - 1)
                            #                 if bit_pos < reg_size:
                            #                     meta['cond_bit'] = clbit_mapping.get((reg_name, bit_pos))
                            # except Exception:
                            #     pass
                        elif isinstance(cond_lhs, (list, tuple)) and len(cond_lhs) > 0:
                            # Some DAGs carry a list/tuple of Clbits for the LHS
                            try:
                                bits = []
                                reg_name = None
                                for b in cond_lhs:
                                    bit_reg = getattr(b, '_register', None)
                                    bit_idx = getattr(b, '_index', None)
                                    if bit_reg is not None and bit_idx is not None:
                                        bits.append(clbit_mapping.get((bit_reg.name, bit_idx)))
                                        # Capture register name if consistent
                                        if reg_name is None:
                                            reg_name = bit_reg.name
                                        elif reg_name != bit_reg.name:
                                            reg_name = None  # mixed registers; drop name
                                meta['cond_val'] = int(cond_val)
                                if reg_name is not None:
                                    meta['cond_register'] = reg_name
                                if len(bits) == 1:
                                    meta['cond_bit'] = bits[0]
                                else:
                                    val = int(cond_val)
                                    if val > 0 and (val & (val - 1)) == 0:
                                        bit_pos = (val.bit_length() - 1)
                                        if bit_pos < len(bits):
                                            meta['cond_bit'] = bits[bit_pos]
                            except Exception:
                                pass
                        else:
                            # Assume a single classical bit with _register and _index
                            bit_reg = getattr(cond_lhs, '_register', None)
                            bit_idx = getattr(cond_lhs, '_index', None)
                            if bit_reg is not None and bit_idx is not None:
                                meta['cond_bit'] = clbit_mapping.get((bit_reg.name, bit_idx))
                            meta['cond_val'] = int(cond_val)
                    except Exception:
                        meta['cond_val'] = None

                # Final fallback: try op.condition_bits if cond_bit not set
                # if 'cond_bit' not in meta:
                #     try:
                #         cond_bits = getattr(node.op, 'condition_bits', None)
                #         cond_tuple = getattr(node.op, 'condition', None)
                #         cond_value = None
                #         if isinstance(cond_tuple, tuple) and len(cond_tuple) == 2:
                #             _, cond_value = cond_tuple
                #         if cond_bits:
                #             mapped = []
                #             for b in cond_bits:
                #                 bit_reg = getattr(b, '_register', None)
                #                 bit_idx = getattr(b, '_index', None)
                #                 if bit_reg is not None and bit_idx is not None:
                #                     mapped.append(clbit_mapping.get((bit_reg.name, bit_idx)))
                #             if len(mapped) == 1:
                #                 meta['cond_bit'] = mapped[0]
                #             elif cond_value is not None:
                #                 v = int(cond_value)
                #                 if v > 0 and (v & (v - 1)) == 0:
                #                     pos = v.bit_length() - 1
                #                     if pos < len(mapped):
                #                         meta['cond_bit'] = mapped[pos]
                #     except Exception:
                #         pass

                gate_info = [gate_name, qubits, registers, params]
                # Only append meta if present to avoid interfering with legacy grouping logic
                if meta:
                    gate_info.append(meta)
                layer_info.append(gate_info)
                if qpu_sizes is not None:
                    if len(qubits) == 2:
                        pairs += 1
                        if pairs >= max_pairs:
                            layer_gates.append(layer_info)
                            layer_info = []
                            pairs = 0
        if layer_info != []:
            layer_gates.append(layer_info)
    return layer_gates

def find_max_interactions(qpu_info):
    max_pairs_qpu = []
    for n in range(len(qpu_info)):
        if qpu_info[n] % 2 == 1:
            max_pairs_qpu.append((qpu_info[n]-1)//2)
        else:
            max_pairs_qpu.append(qpu_info[n]//2)
    max_pairs = sum(max_pairs_qpu)
    return max_pairs

def layer_list_to_dict(layers):
    d = {}
    for i,layer in enumerate(layers):
        d[i] = []
        for gate in layer:
            gate_dict = {}
            name = gate[0]
            qargs = gate[1]
            qregs = gate[2]
            params = gate[3]
            # Extract optional meta dict if present (used for measurement cargs and classical controls)
            meta = gate[4] if (len(gate) >= 5 and isinstance(gate[4], dict)) else {}
            # Skip barriers only; include measurements and resets
            if name == 'barrier':
                continue

            # Preserve legacy 'group' structure: only when len(gate) >= 5 and gate[4] is not a dict
            if len(gate) >= 5 and not isinstance(gate[4], dict):
                gate_dict['type'] = 'group'
                gate_dict['root'] = qargs[0]
                gate_dict['sub-gates'] = []
                gate1 = {}
                gate1['type'] = 'two-qubit'
                gate1['name'] = name
                gate1['time'] = i
                gate1['qargs'] = qargs
                gate1['qregs'] = qregs
                gate1['params'] = params
                gate_dict['sub-gates'].append(gate1)
                for j in range(5,len(gate)):
                    gate_i_list = gate[j]
                    gate_i = {}
                    if gate_i_list[0] == gate_i_list[1]:
                        gate_i['type'] = 'single-qubit'
                        l = 1
                    else:
                        gate_i['type'] = 'two-qubit'
                        l = 2
                    gate_i['name'] = gate_i_list[-1]
                    gate_i['qargs'] = [gate_i_list[0],gate_i_list[1]]
                    gate_i['qregs'] = ['q' for _ in range(l)]
                    gate_i['params'] = gate_i_list[-2]
                    gate_i['time'] = gate_i_list[2]
                    gate_dict['sub-gates'].append(gate_i)
                d[i].append(gate_dict)
                continue

            # Non-group gates (including measure/reset)
            if name == 'measure':
                gate_dict['type'] = 'measure'
                gate_dict['name'] = name
                gate_dict['qargs'] = qargs
                gate_dict['qregs'] = qregs
                gate_dict['params'] = params
                # Record classical bit (if available)
                if meta and 'cargs_idx' in meta and len(meta['cargs_idx']) > 0:
                    gate_dict['cbit'] = meta['cargs_idx'][0]
            else:
                if len(qargs) < 2:
                    gate_dict['type'] = 'single-qubit'
                elif len(qargs) == 2:
                    gate_dict['type'] = 'two-qubit'
                else:
                    gate_dict['type'] = 'multi-qubit'
                gate_dict['name'] = name
                gate_dict['qargs'] = qargs
                gate_dict['qregs'] = qregs
                gate_dict['params'] = params
                # If classically controlled, annotate it (store both register compare and single bit when available)
                if gate_dict.get('type') == 'single-qubit' and meta:
                    if 'cond_register' in meta and 'cond_val' in meta:
                        gate_dict['classical_control_register'] = meta['cond_register']
                        gate_dict['classical_control_val'] = meta['cond_val']
                    if 'cond_bit' in meta:
                        gate_dict['classical_control_bit'] = meta['cond_bit']
                        gate_dict['cbit'] = meta['cond_bit']
            d[i].append(gate_dict)
    return d

