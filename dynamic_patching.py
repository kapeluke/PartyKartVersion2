import jax
import jax.numpy as jnp


""" This file's intention is to be similar to race_selection and race_ordering BUT with one key difference:
the methods in here deal with the case of the total players (P) changing after a certain number of races have already completed.

So most of the logic and solving is similar, but with enough key differences that if it were in the other files it would make them too messy.

I refer to $F$ here as the number of finished races. That is, the race at index $F$ is the "current" race that we can start overwriting.
All races of indices $[0..F-1]$ must be kept constant in ordering and their rosters.
"""

"""Alright since I stalled, let's think this through a bit... how should I tell the user of the necessity of a ghost in a race?
They add a certain number of players. Each of those has two numbers. We can maybe identify if the number goes over or under a minimum boundary.
Maybe we just don't inform the user of any necessity of a ghost or not. I think we have to though. Do we tell them what number specifically causes issues?
Or do we just tell them generally there might be or will be an issue.
"""


# These are the costs of adding a column with the given size. We want races of size 4 and 3
#  and penalize any races of 1 or 2. Change these numbers if you want to affect how the new races
#  are constructed.
RACE_COSTS_BY_SIZE = {
    4: 1.0,
    3: 1.0,
    1: 2.0,
    2: 50.0,
}


def _core_targets(A:jnp.ndarray, F:int) -> jnp.ndarray:
    """Returns the number of races needed for every player in the incidence matrix `A` after `F` races have already completed."""
    N = A.sum(1)[0]
    completed_races = A[:,:F]
    races_needed_for_existing_racers = N - completed_races.sum(1)
    return races_needed_for_existing_racers


# -------------- GHOST RACES THRESHOLD HANDLING --------------
def _ghost_race_add_safe_boundaries(core_targets:jnp.ndarray) -> jnp.ndarray:
    S_core = core_targets.sum()
    N_max_core = core_targets.max()
    safe_max = jnp.floor(S_core / 2)
    safe_min = max(1, 3 * N_max_core - S_core)
    return safe_min, safe_max


def ghost_race_necessity(A:jnp.ndarray, F:int, boundaries:list[tuple[int,int]]) -> tuple[str,list[list[str]]]:
    """Returns information on whether ghost race warnings exist or whether they will absolutely occur or not.

    The first string is global information, and the others specify information on each new player being added and their boundaries' violations.
    """
    num_new = len(boundaries)
    core_targets = _core_targets(A, F)
    safe_min, safe_max = _ghost_race_add_safe_boundaries(core_targets)
    if safe_max > safe_min:
        return ("The remaining core races are too unbalanced. Ghost races are mathematically unavoidable to finish the tournament.", [[]]*num_new)
    warnings:list[list[str]] = []
    for gary_min, gary_max in boundaries: # gary is our fake new player name
        this_player_warnings = []
        if gary_max > safe_max:
            this_player_warnings.append(f"If this player races more than {safe_max} times, Ghost races will be required because the original players do not have enough seats to fill the races.")
        if gary_min < safe_min:
            this_player_warnings.append(f"This player needs to race at least {safe_min} times to help fill out the remaining heats, otherwise Ghost races will be required.")
        warnings.append(this_player_warnings)
    if num_new > 1:
        return ('Multiple new players at once might cause unknown interactions with each other and may cause Ghost races.', warnings)
    else:
        return ('', warnings)


# -------------- ORDER THE NEW RACES --------------


# -------------- ADD AND DROP PLAYER METHODS --------------
def _construct_column_cost_vector(candidates:jnp.ndarray) -> jnp.ndarray:
    """Take the array of incidence's/candidate races and write out the cost of choosing them in a vector."""
    costs = jnp.ones((candidates.shape[0],))
    for size, cost in RACE_COSTS_BY_SIZE.items():
        costs = jnp.where(candidates.sum(1) == size, cost, costs)
    return costs


def _solve_ilp(candidates:jnp.ndarray, lower_bounds_players:jnp.ndarray, upper_bounds_players:jnp.ndarray)


def drop_players(super_pools:jnp.ndarray, A:jnp.ndarray, F:int, dropped_indices:list[int]) -> jnp.ndarray:
    """Handle the dropping of `len(dropped_indices)` from the roster after `F` races."""
    races_needed_for_core_racers = _core_targets(A, F)



def add_players(super_pools:jnp.ndarray, A:jnp.ndarray, F:int, boundaries:list[tuple[int,int]]) -> jnp.ndarray:
    """Handle the addition of `P_new-A.shape[0]` new players to the roster after `F` races."""
    P_old = A.shape[0]
    races_needed_for_core_racers = _core_targets(A, F)
    boundaries_matrix = jnp.array(boundaries)
    lower_bounds_players = jnp.concatenate([races_needed_for_core_racers, boundaries_matrix[:,0]])
    upper_bounds_players = jnp.concatenate([races_needed_for_core_racers, boundaries_matrix[:,1]])


if __name__ == "__main__":
    A = jnp.array([[0, 1, 1, 1],[0, 1, 1, 1],[1, 1, 0, 1],[1, 0, 1, 1],[1, 1, 1, 0]])
    A_new = add_players(A, F=2, boundaries=[(1,3),(1,3)])
    print(A_new)
