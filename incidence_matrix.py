import jax
import jax.numpy as jnp


P = 7 # number of players
N = 3 # number of races each player races in

R = int(jnp.ceil(P*N/4)) # total number of races

# element m[i,j]=1 if player i is in race j, 0 otherwise
# columns should sum to 4 or 3
# rows must sum to N
m = jnp.array(
    [[1, 0, 0, 1, 0, 1],
     [1, 1, 0, 1, 0, 0],
     [1, 0, 1, 0, 1, 0],
     [0, 1, 1, 0, 1, 0],
     [0, 1, 0, 1, 1, 0],
     [1, 0, 1, 0, 0, 1],
     [0, 1, 1, 0, 0, 1]], dtype=jnp.float32
)
assert m.shape == (P,R)

# co-occurance matrix, C:
# diagonal elements C[i,i] equals the total races player i has played (should be N)
# off-diagonal elements C[i,j] (i≠j) is the exact number of times pairing (i,j) has occurred
C = m @ m.T

print(C)

# want to reduce the variance of the off-diagonal elements
def race_selection_loss(c):
    mask = ~jnp.eye(c.shape[0], dtype=bool)
    off_diagonal_elements = c[mask]
    return jnp.var(off_diagonal_elements)

print(race_selection_loss(C))