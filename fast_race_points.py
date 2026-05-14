from typing import Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp


P = 7
R = 7
TP = P*P
TP_COND = (P-2)*(P-2)


def all_pairs():
    return jnp.zeros((P,P-1), dtype=jnp.int8)

def select_pair(rng):
    # TODO -- broken
    idx = jax.random.randint(rng, (), 0, P*(P-1))
    print(idx)
    return (idx // P, idx % P)

def conditional_select_pair(rng, c):
    # TODO -- broken
    idx = jax.random.randint(rng, (), 0, TP_COND)
    p = jnp.array([idx // (P-2), idx % (P-2)])
    p = jnp.where(p[0] >= c[0], p + jnp.array([1,0]), p)
    p = jnp.where(p[1] >= c[1], p + jnp.array([0,1]), p)
    return tuple(p)

def count_pair(m, l):
    # TODO -- broken.
    ret = m.at[l].set(m[l] + 1)
    reverse_l = (l[1],l[0])
    ret = ret.at[reverse_l].set(ret[reverse_l] + 1)
    return ret


SEED = 1
rng = jax.random.key(SEED)

m = all_pairs()
p = select_pair(rng)
m = count_pair(m, p)
p2 = conditional_select_pair(rng, p)
print(count_pair(m, p2))

m = all_pairs()
p = select_pair(rng); print(f"{p=}")
m = count_pair(m, p)
for _ in range(1000):
    rng, _ = jax.random.split(rng)
    p2 = conditional_select_pair(rng, p)
    m2 = count_pair(m, p2)
    assert (m2 == 2).any() == False, f"{p2=} with {rng=}"
print('good')