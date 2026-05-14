import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import milp, Bounds, LinearConstraint
from itertools import combinations
from typing import Optional


from timer import Timer
TIMER = Timer()


P = 17
N = 3

NUM_BEST_STENCILS = 60
GAP_VARIANCE_PENALTY_WEIGHT = 0.1 # how much emphasis is put on gap variance when choosing the best stencils
CUTS_PER_ROTATION = 4 # means that our batch per "best stencil" is of length/size (CUTS_PER_ROTATION+1)*P
STRICT_R = False # Do we want ceil(NP/4) to be enforced strictly or just kind of?

assert CUTS_PER_ROTATION <= 4, f"CUTS_PER_ROTATION can be at most 4 (since 4 people per stencil)"
assert NUM_BEST_STENCILS <= 5*P, f"NUM_BEST_STENCILS can be at most {5*P} since there are only that many in total."


def choose_optimal_stencils(P:int, num_best:int, gap_var_weight:float) -> jnp.ndarray:
    """
    Evaluates all cyclic base blocks and returns the top `num_best` stencils
    based on difference structure, modular dispersion, and gap variance.
    """
    # 1. Generate all combinations of 3 elements from 1 to P-1
    sub_combos = list(combinations(range(1, P), 3))
    # Shape: (len(sub_combos), 4). Base holes always start with 0.
    holes = jnp.array([[0] + list(c) for c in sub_combos])
    # METRICS 1 & 2: Difference Structure & Modular Dispersion
    # Calculate all pairwise differences
    diffs = (holes[:, :, None] - holes[:, None, :]) % P  # Shape: (N, 4, 4)
    # Map to shortest chords (e.g., if P=17, a difference of 12 becomes 5)
    chords = jnp.minimum(diffs, P - diffs)
    # Count frequencies of each chord length up to max possible chord
    max_chord = P // 2
    chord_counts = jax.nn.one_hot(chords, max_chord + 1).sum(axis=(1, 2))
    # Ignore the 0-chord (the diagonal of the subtraction grid)
    chord_counts_nonzero = chord_counts[:, 1:]
    # Sum of Squares penalty: enforces uniform coverage and penalizes resonance
    chord_penalty = jnp.sum(chord_counts_nonzero ** 2, axis=1)
    # METRIC 3: Gap Variance
    # Calculate adjacent physical gaps within the block
    internal_gaps = jnp.diff(holes, axis=1)
    # Calculate the cyclic wrap-around gap from the last to the first element
    wrap_around = (P - holes[:, -1:] + holes[:, 0:1])
    # Combine to get all 4 gaps for each block. Shape: (N, 4)
    all_gaps = jnp.concatenate([internal_gaps, wrap_around], axis=1)
    # Calculate the variance of these adjacent gaps
    gap_var = jnp.var(all_gaps, axis=1)
    # COMBINE SCORES & SELECT WINNERS
    # Chord penalty dictates the dominant tier of mathematical fairness.
    # We apply gap variance with a small weight to break ties and penalize 
    # blocks with extreme physical clumping (e.g., huge empty voids in the circle).
    total_scores = chord_penalty + (gap_var_weight * gap_var)
    # Get the indices of the lowest N scores
    best_indices = jnp.argsort(total_scores)[:num_best]
    # Extract the winning blocks
    best_blocks = holes[best_indices]
    # Convert absolute player positions back into shift intervals (stencils)
    optimal_stencils = jnp.diff(best_blocks, axis=1)
    return optimal_stencils # shape=(num_best,3)


def choose_best_stencil(P:int):
    return choose_optimal_stencils(P, 1, GAP_VARIANCE_PENALTY_WEIGHT)[0,:]


def generate_batch(P:int, stencil:jnp.ndarray, cuts_per_rotation:int) -> tuple[jnp.ndarray, int]:
    """Generates the incidence matrix purely functionally using JAX one-hot encoding."""
    # 1. Generate the base size-4 holes
    shifts = jnp.concatenate([jnp.array([0]), stencil])
    base_holes = jnp.cumsum(shifts) % P
    # 2. Generate all P rotations simultaneously via broadcasting
    rotations = jnp.arange(P)[:, None]
    size_4_holes = (base_holes + rotations) % P
    # 3. Create the incidence matrix for size-4 races (Functional, no in-place edits)
    # one_hot makes it (P, 4, P). Summing axis 1 flattens the 4 players into a single row of length P.
    # We transpose (.T) so that Columns = Races and Rows = Players.
    inc_4 = jax.nn.one_hot(size_4_holes, P, dtype=jnp.int32).sum(axis=1).T
    # 4. Generate size-3 matrices by subtracting players
    inc_3_list = []
    for i in range(cuts_per_rotation):
        # Isolate the exact players we want to drop across all P races
        dropped_players = size_4_holes[:, i]
        # Convert those dropped players into their own incidence matrix
        dropped_inc = jax.nn.one_hot(dropped_players, P, dtype=jnp.int32).T
        # Functionally subtract the dropped player from the size-4 races!
        inc_3_list.append(inc_4 - dropped_inc)
    # 5. Concatenate everything together along the columns
    incidence_matrix = jnp.concatenate([inc_4] + inc_3_list, axis=1)
    num_candidates = incidence_matrix.shape[1]
    return incidence_matrix, num_candidates


def solve_ilp_relaxed(P:int, N:int, incidence_matrix:jnp.ndarray, num_candidates:int) -> Optional[jnp.ndarray]:
    # just specifying the shape of our input here I think
    c = np.ones(num_candidates)
    # want incidence_matrix@x=N*1, so our lower and upper bounds is exactly N*1 (target_races)
    target_races = np.full(P, N)
    constraints = LinearConstraint(incidence_matrix, lb=target_races, ub=target_races)
    # tell them we want the solution to be either 0 (lower) or 1 (upper) bound for x
    bounds = Bounds(np.zeros(num_candidates), np.ones(num_candidates))
    # say it's all integers, no fractions. 1 means integer, 0 means continuous
    integrality = np.ones(num_candidates)
    # print(f"Solving ILP with {P=}, {N=}, {num_candidates=} ...")
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if result.success:
        # print("\nOptimal schedule found!")
        chosen_races_mask = np.round(result.x).astype(int)
        selected_indices = np.where(chosen_races_mask == 1)[0]
        # print(f"Total races scheduled = {len(selected_indices)}")
        # print(f"Selected candidate indices = {selected_indices}")
        final_schedule_matrix = incidence_matrix[:,selected_indices]
        # races_per_player = np.sum(final_schedule_matrix, axis=1)
        # print(f"\nRaces per player (should all be {N}) = {races_per_player}")
        return final_schedule_matrix
    else:
        # print("\nNo mathematical solution exists for this specific candidate pool.")
        # print("Try a different starting stencil or expand the pool.")
        return None


def solve_ilp_strict(P:int, N:int, incidence_matrix:jnp.ndarray, num_candidates:int) -> Optional[jnp.ndarray]:
    R = jnp.ceil(P * N / 4)
    # want incidence_matrix@x=N*1, so our lower and upper bounds is exactly N*1 (target_races)
    # Create row of all 1s for the total race constraint
    race_count_row = jnp.ones((1,num_candidates))
    # stack it below the original incidence_matrix. Shape is now (P+1, num_candidates)
    A_constrainted = jnp.vstack([incidence_matrix, race_count_row])
    # The bounds for the P players is N. The bound for the last row is R
    lower_bounds = jnp.append(jnp.full(P, N), R)
    upper_bounds = lower_bounds.copy()
    constraints = LinearConstraint(A_constrainted, lb=lower_bounds, ub=upper_bounds)
    # Objective doesn't matter anymore since R is fixed, but we keep it
    c = np.ones(num_candidates)
    bounds = Bounds(np.zeros(num_candidates), np.ones(num_candidates))
    integrality = np.ones(num_candidates)
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if result.success:
        # print("\nOptimal schedule found!")
        chosen_races_mask = np.round(result.x).astype(int)
        selected_indices = np.where(chosen_races_mask == 1)[0]
        # print(f"Total races scheduled = {len(selected_indices)}")
        # print(f"Selected candidate indices = {selected_indices}")
        final_schedule_matrix = incidence_matrix[:,selected_indices]
        # races_per_player = np.sum(final_schedule_matrix, axis=1)
        # print(f"\nRaces per player (should all be {N}) = {races_per_player}")
        return final_schedule_matrix
    else:
        # print("\nNo mathematical solution exists for this specific candidate pool.")
        # print("Try a different starting stencil or expand the pool.")
        return None


def get_var_incidence_matrix(A_sel:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Produce the co-occurrence matrix:
    #  diagonal elements C[i,i] equals the total races player i has played (should be N)
    #  off-diagonal elements C[i,j] (i≠j) is the exact number of times pairing (i,j) has occurred
    co_occurrence_matrix = A_sel @ A_sel.T
    # Get the off-diagonal elements
    mask = ~jnp.eye(co_occurrence_matrix.shape[0], dtype=bool)
    off_diagonal_elements = co_occurrence_matrix[mask]
    # Return the variance of the off-diagonal elements
    return co_occurrence_matrix, jnp.var(off_diagonal_elements)


def get_best_unordered_races(P:int, N:int, strict_R:bool=STRICT_R, cuts_per_rotation:int=CUTS_PER_ROTATION, num_stencils:int=NUM_BEST_STENCILS, gap_variance_weight:float=GAP_VARIANCE_PENALTY_WEIGHT):
    solve_ilp = solve_ilp_strict if strict_R else solve_ilp_relaxed
    with TIMER.time('choose_optimal_stencils'):
        best_stencils = choose_optimal_stencils(P, num_stencils, gap_variance_weight)
    lowest_stencil_variance = jnp.inf
    best_selected_incidence_matrix = None
    best_co_occurrence_matrix = None
    for stencil_idx in range(best_stencils.shape[0]):
        stencil = best_stencils[stencil_idx,:]
        print(f"Working on stencil #{stencil_idx} = {stencil}")
        with TIMER.time('generate_batch'):
            A, num_candidates = generate_batch(P, stencil, cuts_per_rotation)
        with TIMER.time('solve_ilp'):
            A_sel = solve_ilp(P, N, A, num_candidates)
        if A_sel is not None:
            with TIMER.time('get_var_incidence_matrix'):
                Cmtx, var = get_var_incidence_matrix(A_sel)
            if var < lowest_stencil_variance:
                best_selected_incidence_matrix = A_sel
                lowest_stencil_variance = var
                best_co_occurrence_matrix = Cmtx
    if A_sel is None:
        print(f"Something went wrong. Didn't find a good solution at all.")
    return best_selected_incidence_matrix, lowest_stencil_variance, best_co_occurrence_matrix


if __name__ == "__main__":
    methods = ['choose_optimal_stencils', 'generate_batch', 'solve_ilp', 'get_var_incidence_matrix']

    with TIMER.time('total_runtime'):
        A, variance, C = get_best_unordered_races(P, N)
    # print(A)
    # print(A.sum(0))
    # print(C)
    # print(f"{variance=}")

    TIMER.nice_summary()