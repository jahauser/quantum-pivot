# coding=utf8

import numpy as np
import tools

phase_gate = lambda phi : np.array([[1, 0],
                                    [0, np.exp(1j*phi)]])

gate_matrices= {"H": 1.0 / 2 ** 0.5 * np.array([[1, 1],
                                                [1, -1]]),
                "T": tools.phase_gate(np.pi / 4),
                "S": tools.phase_gate(np.pi / 2),
                "CNOT": np.tensordot(tools.P0, np.eye(2), axes=0) +
                        np.tensordot(tools.P1, tools.pauli_X, axes=0),
                "P0": tools.P0,
                "P1": tools.P1,
                "X": tools.pauli_X,
                "Y": tools.pauli_Y,
                "NOT": np.array([[0, 1],
                                 [1, 0]])}



def rand_gates(gate_set, N, num_gates):
    k = 0
    while k < num_gates:
        gate = np.random.choice(gate_set)
        matrix = gate_matrices[gate]
        dim = int(len(matrix.shape) / 2)
        yield Gate(matrix, np.random.choice(np.arange(N), size=dim, replace=False), dim=dim, name=gate)
        k += 1



class Gate:
    def __init__(self, matrix, bits, dim=None, name=None):
        self.matrix = matrix
        self.bits = bits
        if not dim:
            self.dim = int(len(self.matrix.shape)/2)
        else:
            self.dim = dim

        self.name = name


    #@profile
    def apply_to(self, state):
        my_comps = [2*k+1 for k in range(self.dim)]

        N = len(state.shape)
        transposition = []
        index = {bit: num for (num, bit) in enumerate(self.bits)}
        base = list(range(len(self.bits),N))
        for i in range(N):
            contained = i in self.bits
            if contained:
                transposition.append(index[i])
            else:
                transposition.append(base.pop(0))

        td = np.tensordot(self.matrix, state, axes=(my_comps, self.bits))
        return np.transpose(td,transposition)

    def dagger(self):
        if self.name[-1] == '†':
            new_name = self.name[:-1]
        else:
            new_name = self.name + '†'
        transposition = []
        for i in range(self.dim):
            transposition.extend([2*i+1, 2*i])
        return Gate(np.transpose(self.matrix.conjugate(), transposition), self.bits, dim=self.dim, name=new_name)

    def __str__(self):
        if self.name:
            return self.name + str(self.bits)
        return self.matrix

    def __repr__(self):
        if self.name:
            return self.name + str(self.bits)
        return self.matrix.__repr__()


class GateSequence(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def apply_to(self, state):
        for gate in self:
            state = gate.apply_to(state)
        return state

    def apply_to2(self, state):
        if not self:
            return state
        return self[1:].apply_to(self[0].apply_to(state))

    def dagger(self):
        return GateSequence(gate.dagger() for gate in self[::-1])

    def __getitem__(self, *args):
        if isinstance(args[0], int):
            return list.__getitem__(self, *args)
        return GateSequence(list.__getitem__(self, *args))

    def __add__(self, other):
        return GateSequence(list.__add__(self, other))
