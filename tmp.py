import numpy as np

combinations = []
for i in np.arange(0.0, 1.1, 0.1):
    combinations.append(i)

from itertools import permutations

perms = permutations(combinations, 3)
first = False
for perm in perms:
    if round(perm[0], 1) == 0.3 and round(perm[1], 1) > 0.3 and not first:
        print('"0.33 0.33 0.33"')
        first = True
    if sum(perm) == 1.0:
        print(f'"{round(perm[0], 2)} {round(perm[1], 2)} {round(perm[2], 2)}"')
