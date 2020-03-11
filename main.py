# coding=utf8

import numpy as np
import tools
import gates
from gates import GateSequence


clifford_set = ["CNOT", "H", "S"]
universal_set = ["CNOT", "H", "T"]


def sandwich_maker(N, mU, mT):
    U1 = GateSequence(gates.rand_gates(clifford_set, N, mU))
    U2 = GateSequence(gates.rand_gates(clifford_set, N, mU))
    T = GateSequence(gates.rand_gates(["T"], N, mT))

    return U1 + T + U2


def ptrace_outer(u, keep, dims, optimize=False):
    """Calculate the partial trace of an outer product
    
    (source: https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python)

    ρ_a = Tr_b(|u><u|)

    Parameters
    ----------
    u : array
        Vector to use for outer product
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    u = u.reshape(dims)
    rho_a = np.einsum(u, idx1, u.conj(), idx2)#, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)

def r_twiddle(N, state):
    # deal with contiguousness later
    state = state.reshape(2**N)
    traced = ptrace_outer(state, list(range(0,int(N/2))), [2 for _ in range(N)])

    p_k = np.linalg.svd(traced, compute_uv=False)#, hermitian=True)

    l_k = -np.sort(-np.sqrt(p_k))
    rt_k = []
    for k in range(0,len(l_k)-1):
        d1 = 0
        d2 = 0
        if k > 0:
            d1 = l_k[k-1]-l_k[k]
        if k < len(l_k) - 1:
            d2 = l_k[k] - l_k[k+1]
        if max([d1, d2]) != 0:
            rt_k.append(min([d1, d2])/max([d1, d2]))

    return np.mean(rt_k)

def sample(N, arch, arch_params): # maybe also architecture?
    state = tools.tensorprod(*[tools.zero for n in range(N)])
    #state = tools.tensorprod(*tools.rand_states(N))

    if arch == "sandwich":
        circuit = sandwich_maker(*arch_params)

    state = circuit.apply_to(state)

    rt = r_twiddle(N, state)

    return rt

def main():
    pass


if __name__ == '__main__':
    print(np.mean([sample(12, "sandwich", [8, 100, 100]) for _ in range(100)]))