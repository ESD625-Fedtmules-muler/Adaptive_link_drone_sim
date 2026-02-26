from numba import njit

@njit
def test(x):
    return x + 1

print(test(5))