import numpy as np
import time

zero = np.array([1.0, 0.0])
one  = np.array([0.0, 1.0])

P0 = np.outer(zero, zero)
P1 = np.outer(one, one)

pauli_X = np.array([[0, 1],
                    [1, 0]])

pauli_Y = np.array([[0,-1j],
                    [1j,0]])

phase_gate = lambda phi : np.array([[1, 0],
                                    [0, np.exp(1j*phi)]])

times = {} # Add logging
def timeit(label=None):
    def wrap(method):
        def timed_f(*args, **kwargs):
            ts = time.time()
            result = method(*args, **kwargs)
            te = time.time()

            if label:
                times[label] = times.get(label, 0) + te-ts
            else:
                print(te-ts)

            return result
        return timed_f
    return wrap


# consider func_tools
def tensorprod(head, *rest):
    if not rest:
        return head
    return np.tensordot(head, tensorprod(*rest), axes=0)


def rand_states(N):
    i = 0
    while i < N:
        theta = np.pi * np.random.uniform()
        phi = 2 * np.pi * np.random.uniform()
        yield np.cos(theta/2)*zero + np.sin(theta/2) * np.exp(1j*phi)*one
        i += 1

def inner_product(state1, state2):
    N = len(state1.shape)
    return np.tensordot(state1.conjugate(), state2, axes=(list(range(N)), list(range(N))))