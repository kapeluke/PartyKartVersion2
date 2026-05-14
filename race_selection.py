import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix
from itertools import combinations
from typing import Optional

import math


"""This file's goal is to solve the following problem:

How do we select an unordered set of races such that the variance of the count of every pairwise connection between players
is as little as possible; given `P` players and `N` races strictly for each player. We can also choose to minimize the
total number of races for the entire tournament with some leniency. The maximum size per race is 4, and the minimum is 3.

This is generally being accomplished within this file through the use of patterns called "stencils". Imagine your `P` players
standing evenly-spaced around a circle. Each player has a number in [0..P-1]. If you place a piece of solid cardboard over the entire
circle, and cut out 3 or 4 holes, you determine the roster in a race with the numbers you can see through the cardboard.
Now, if we imagine fixing the first hole at player 0, we then only need to select 3 "offsets" to select the other 3 players in a race.
For example, picking the stencil/offsets (2,4,5) yields players: holes=(0, 0+2 mod P, 0+2+4 mod P, 0+2+4+5 mod P). To generate `P` races,
we just cyclically rotate the stencil around the circle. So the next race's schedule is `holes + (1,1,1,1) mod P`. Not all of these holes
or stencils will create good races though: if we chose a stencil with offsets (1,1,1) then every race will only contain 4 subsequent races.
If we have 17 people (P=17) then we'd never have a race of player 0 against player 11, for example. This is bad design. So now our search
space (for the most part) just simply becomes choosing the 3 offsets such that we have a good schedule when rotating the stencil around.

To handle the case of some combinations of (P,N) needing races of size 3, we just take some of our size=4 stencils and patch back one of the
holes to make it a three person race.

There's unfortunately another problem. If we have a P*N sufficiently large, no matter what stencil we choose, there will ALWAYS be pairwise
connections / chords that are impossible to have together. Each 4-person race only creates 6 pairwise connectoins and each 3-person race
only creates 3 pairwise connections. So even if we have P=17 and N=5 yielding ceil(P*N/4)=ceil(17*5/4)=22 races, we only have space for
22*6=132 pairwise connections total. HOWEVER, there are nCr(17,2)=136 total pairwise connections to cover. Our cyclic structure will
actually not even create 132 unique pairwise connections, but because its the same stencil just rotated it will omit some chord lengths
like 4: 0&4 or 1&5 or 2&6. So to allow our solver to spread out some of these doubled-up pairwise connections, we create a "super-pool"
of stencils for the solver to choose from. So now, rather than using one stencil and just choosing the best of that, we give it a few
stencils in a "super-pool". The super-pool is created such that each stencil chosen complements its other pool members' missing chord lengths.

All of this sounds complicated, and it is. Sometimes the solver gets stuck on a solution and can't find a better one, even if it exists...
yes computers are funny like that. So we add some random noise in by shuffling the proposed rosters to begin with, which will let the solver
look at some things first vs. others at random. We also give the solver an upper time limit and the ability to turn down the "strictness"
of the solver's solutions. The strictness here refers to how close we want the solver to be to the minimum number of races, R=ceil(P*N/4).
The more strict we are, the higher the variance gets (worse solutions), but the tournament becomes shorter.
"""


P = 17
N = 3

NUM_BEST_STENCILS = 60
GAP_VARIANCE_PENALTY_WEIGHT = 0.1 # how much emphasis is put on gap variance when choosing the best stencils
CUTS_PER_ROTATION = 4 # means that our batch per "best stencil" is of length/size (CUTS_PER_ROTATION+1)*P
STRICT_R = False # Do we want ceil(NP/4) to be enforced strictly or just kind of?
SUPERPOOL_SIZE = 3 # The size of the superpool of stencil rotations
NUM_RESTARTS = 10 # Number of times we try to restart with new noise
CEILING_INCREASE_LIMIT = 3 # Number of times we're allowed to increase the max pairwise limit before quitting
ILP_MAX_TIME_LIMIT = 5.0 # seconds

assert CUTS_PER_ROTATION <= 4, f"CUTS_PER_ROTATION can be at most 4 (since 4 people per stencil)"
assert NUM_BEST_STENCILS <= 5*P, f"NUM_BEST_STENCILS can be at most {5*P} since there are only that many in total."


def evaluate_all_stencils(P:int, gap_var_weight:float=GAP_VARIANCE_PENALTY_WEIGHT):
    """
    Evaluates all cyclic base blocks and returns them sorted by base score, along with their exact chord coverages so we can mix and match them later.
    """
    sub_combos = list(combinations(range(1, P), 3)) # chord combinations / stencils
    holes = jnp.array([[0] + list(c) for c in sub_combos]) # holes
    # Calculate chords and their frequencies
    diffs = (holes[:, :, None] - holes[:, None, :]) % P # chord lengths
    chords = jnp.minimum(diffs, P - diffs) # chord lengths but consider that they can wrap around again
    max_chord = P // 2 # max chord length
    chord_counts = jax.nn.one_hot(chords, max_chord + 1).sum(axis=(1, 2))
    chord_counts_nonzero = chord_counts[:, 1:]
    # Base Penalties
    chord_penalty = jnp.sum(chord_counts_nonzero ** 2, axis=1)
    internal_gaps = jnp.diff(holes, axis=1)
    wrap_around = (P - holes[:, -1:] + holes[:, 0:1])
    all_gaps = jnp.concatenate([internal_gaps, wrap_around], axis=1)
    gap_var = jnp.var(all_gaps, axis=1)
    total_scores = chord_penalty + (gap_var_weight * gap_var)
    stencils = jnp.diff(holes, axis=1)
    # Sort all arrays so the fundamentally best stencils are at index 0
    sort_idx = jnp.argsort(total_scores)
    return stencils[sort_idx], total_scores[sort_idx], chord_counts_nonzero[sort_idx]


def build_complementary_chunks(stencils, scores, chord_coverages, num_chunks, pool_size):
    """
    Greedily builds chunks of size `pool_size` by vectorizing the search for the stencil that best 'flattens' the current pool's chord coverage.
    """
    N_total = len(stencils)
    # penalty_modifiers ensures we never pick the same stencil twice.
    # We set chosen indices to jnp.inf.
    penalty_modifiers = jnp.zeros(N_total)
    chunks = []
    for _ in range(num_chunks):
        current_chunk = []
        # Track the cumulative chord coverage of the stencils currently in the pool
        current_pool_coverage = jnp.zeros(chord_coverages.shape[1])
        for _ in range(pool_size):
            # VECTORIZED MAGIC: Add the current pool's coverage to ALL available stencils simultaneously
            proposed_coverages = current_pool_coverage + chord_coverages
            # The Sum of Squares of the proposed coverages. 
            # This mathematically punishes peaks and aggressively rewards filling in 0s.
            pool_chord_penalties = jnp.sum(proposed_coverages ** 2, axis=1)
            # We add a fraction of the base score to act as a tie-breaker, 
            # ensuring we prefer structurally sound stencils to fill the gaps.
            total_penalties = pool_chord_penalties + (0.5 * scores) + penalty_modifiers
            # Grab the index of the absolute best complementary stencil
            best_idx = int(jnp.argmin(total_penalties))
            # Add it to the chunk and update the pool's coverage
            current_chunk.append(stencils[best_idx])
            current_pool_coverage += chord_coverages[best_idx]
            # Mask out this stencil so it can't be used again
            penalty_modifiers = penalty_modifiers.at[best_idx].set(jnp.inf)
        chunks.append(jnp.stack(current_chunk))
    return jnp.stack(chunks) # Shape: (num_chunks, pool_size, 3)


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


def calculate_theoretical_min_overlap(P:int, N:int) -> int:
    """Calculates the absolute mathematical floor for max overlap."""
    total_slots = P * N
    # Calculate how many 4-player and 3-player races we expect
    races_4_slots = total_slots // 4
    races_3_slots = total_slots % 4
    # If there are leftover slots, they form a smaller race (3)
    pairs_from_4s = races_4_slots * 6
    pairs_from_leftovers = math.comb(races_3_slots, 2) if races_3_slots > 1 else 0
    total_pairs_generated = pairs_from_4s + pairs_from_leftovers
    total_unique_pairs = math.comb(P, 2)
    # The average overlap dictates the absolute minimum ceiling
    average_overlap = total_pairs_generated / total_unique_pairs
    return math.ceil(average_overlap)


def solve_ilp_relaxed(P:int, N:int, incidence_matrix:jnp.ndarray, num_candidates:int, shuffle_seed:Optional[int]=42, max_overlap:int=2) -> Optional[jnp.ndarray]:
    # Column shuffling
    if shuffle_seed is not None:
        key = jax.random.key(shuffle_seed)
        shuffle_idx = jax.random.permutation(key, num_candidates)
    else:
        shuffle_idx = jnp.arange(num_candidates)
    # Shuffle the candidate races. 
    shuffled_incidence = incidence_matrix[:, shuffle_idx]
    # Pure integer objective. Allows massive B&B tree pruning!
    c = np.ones(num_candidates) 
    # 1. Base constraints (N races per player)
    target_races = jnp.full(P, N)
    # 2. Pairwise Co-occurrence Constraints
    i_idx, j_idx = jnp.triu_indices(P, k=1)
    # Build pairwise overlaps using the SHUFFLED matrix
    pairwise_overlaps = shuffled_incidence[i_idx] * shuffled_incidence[j_idx]
    # Stack constraints
    A_constrained = jnp.vstack([shuffled_incidence, pairwise_overlaps])
    lower_bounds = jnp.append(target_races, jnp.zeros(len(i_idx)))
    upper_bounds = jnp.append(target_races, jnp.full(len(i_idx), max_overlap))
    # Convert the dense JAX array to a NumPy array, then to a SciPy CSC sparse matrix
    A_sparse = csc_matrix(np.array(A_constrained))
    constraints = LinearConstraint(A_sparse, lb=lower_bounds, ub=upper_bounds)
    bounds = Bounds(np.zeros(num_candidates), np.ones(num_candidates))
    integrality = np.ones(num_candidates)
    options = {'time_limit': ILP_MAX_TIME_LIMIT} 
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
    if result.success and result.x is not None:
        chosen_races_mask = np.round(result.x).astype(int)
        selected_shuffled_indices = np.where(chosen_races_mask == 1)[0]
        # UN-SHUFFLE: Map the selected indices back to their original positions
        selected_original_indices = shuffle_idx[selected_shuffled_indices]
        return incidence_matrix[:, selected_original_indices]
    else:
        return None


def solve_ilp_strict(P:int, N:int, incidence_matrix:jnp.ndarray, num_candidates:int, shuffle_seed:Optional[int]=42, max_overlap:int=2, max_extra_races:int=0) -> Optional[jnp.ndarray]:
    # Calculate the absolute mathematical minimum number of races needed
    R = int(jnp.ceil(P * N / 4))
    if shuffle_seed is not None:
        key = jax.random.key(shuffle_seed)
        shuffle_idx = jax.random.permutation(key, num_candidates)
    else:
        shuffle_idx = jnp.arange(num_candidates)
    shuffled_incidence = incidence_matrix[:, shuffle_idx]
    # Pure integer objective. Allows massive B&B tree pruning!
    c = np.ones(num_candidates) 
    # 1. Base constraints (N races per player)
    target_races = jnp.full(P, N)
    # 2. Pairwise Co-occurrence Constraints
    i_idx, j_idx = jnp.triu_indices(P, k=1)
    pairwise_overlaps = shuffled_incidence[i_idx] * shuffled_incidence[j_idx]
    # 3. Total Races Constraint (The "Strict" condition)
    race_count_row = jnp.ones((1, num_candidates))
    # Stack constraints
    A_constrained = jnp.vstack([shuffled_incidence, pairwise_overlaps, race_count_row])
    # Concatenate bounds: Player targets -> Pairwise overlaps -> Total races (R)
    lower_bounds = jnp.concatenate([
        target_races,                      # Players must race exactly N times
        jnp.zeros(len(i_idx)),             # Overlaps min is 0
        jnp.array([R])                     # Total races must be at least R
    ])
    upper_bounds = jnp.concatenate([
        target_races,                      # Players must race exactly N times
        jnp.full(len(i_idx), max_overlap), # Overlaps max limit
        jnp.array([R + max_extra_races])   # Total races can be up to R + tolerance
    ])
    A_sparse = csc_matrix(np.array(A_constrained))
    constraints = LinearConstraint(A_sparse, lb=lower_bounds, ub=upper_bounds)
    bounds = Bounds(np.zeros(num_candidates), np.ones(num_candidates))
    integrality = np.ones(num_candidates)
    options = {'time_limit': ILP_MAX_TIME_LIMIT} 
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
    if result.success and result.x is not None:
        chosen_races_mask = np.round(result.x).astype(int)
        selected_shuffled_indices = np.where(chosen_races_mask == 1)[0]
        # UN-SHUFFLE: Map the selected indices back to their original positions
        selected_original_indices = shuffle_idx[selected_shuffled_indices]
        return incidence_matrix[:, selected_original_indices]
    else:
        return None


def solve_ilp_auto_relaxing(P:int, N:int, incidence_matrix:jnp.ndarray, num_candidates:int, strict_R:bool=STRICT_R, strictness_tolerance:int=0, shuffle_seed:Optional[int]=42):
    """
    Attempts to solve the ILP at the mathematical limit. 
    If it fails, it relaxes the max_overlap bound by 1 until it finds a solution.
    """
    # solve_ilp = solve_ilp_strict if strict_R else solve_ilp_relaxed
    # Start at the absolute mathematical floor
    current_max_overlap = calculate_theoretical_min_overlap(P, N)
    # We will allow it to relax up to a highly permissive bound if necessary
    absolute_ceiling = current_max_overlap + CEILING_INCREASE_LIMIT
    while current_max_overlap <= absolute_ceiling:
        # print(f"\tAttempting to solve with max_overlap <= {current_max_overlap}...")
        # Call your modified ILP function (from previous step) 
        # that includes the pairwise overlap constraints
        if strict_R:
            A_sel = solve_ilp_strict(
                P, N, incidence_matrix, num_candidates, 
                shuffle_seed=shuffle_seed, 
                max_overlap=current_max_overlap,
                max_extra_races=strictness_tolerance
            )
        else:
            A_sel = solve_ilp_relaxed(
                P, N, incidence_matrix, num_candidates, 
                shuffle_seed=shuffle_seed, 
                max_overlap=current_max_overlap
            )
        if A_sel is not None:
            # print(f"\tSuccess! Schedule found with max overlap {current_max_overlap}")
            return A_sel   
        # If the ILP failed, the stencils couldn't achieve this bound.
        # Relax the constraint and try again.
        current_max_overlap += 1
    print("\tFailed to find a schedule even after relaxing bounds.")
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


def get_best_unordered_races(P:int, N:int,
                             strict_R:bool=STRICT_R,
                             strictness_tolerance:int=0,
                             cuts_per_rotation:int=CUTS_PER_ROTATION,
                             num_stencils:int=NUM_BEST_STENCILS,
                             gap_variance_weight:float=GAP_VARIANCE_PENALTY_WEIGHT,
                             pool_size:int=SUPERPOOL_SIZE,
                             restarts:int=NUM_RESTARTS,
                             seed:Optional[int]=42):
    # 1. Get all evaluated stencils
    all_stencils, all_scores, all_coverages = evaluate_all_stencils(P, gap_variance_weight)
    # 2. Build perfectly complementary chunks
    #   This means that where some chords lack in some stencils, we want to select other stencils that make up for their missing chord lengths
    num_chunks = max(1, num_stencils // pool_size)
    super_pools = build_complementary_chunks(all_stencils, all_scores, all_coverages, num_chunks, pool_size)
    # Find the best incidence matrix
    lowest_stencil_variance = jnp.inf
    best_selected_incidence_matrix = None
    best_co_occurrence_matrix = None
    # 3. Iterate over the curated super-pools
    for chunk_idx in range(super_pools.shape[0]):
        chunk = super_pools[chunk_idx]
        print(f"Working on Complementary Super-Pool #{chunk_idx} ...")
        matrices = []
        for i in range(pool_size):
            A_part, _ = generate_batch(P, chunk[i], cuts_per_rotation)
            matrices.append(A_part)
        print(f"\tGenerated batches.")
        A_super = jnp.concatenate(matrices, axis=1)
        num_candidates = A_super.shape[1]
        for restart in range(restarts):
            A_sel = solve_ilp_auto_relaxing(P, N, A_super, num_candidates, strict_R=strict_R, strictness_tolerance=strictness_tolerance, shuffle_seed=seed+restart)
            if A_sel is not None:
                Cmtx, var = get_var_incidence_matrix(A_sel)
                if var < lowest_stencil_variance:
                    best_selected_incidence_matrix = A_sel
                    lowest_stencil_variance = var
                    best_co_occurrence_matrix = Cmtx
    if best_selected_incidence_matrix is None:
        print("Something went wrong. Didn't find a good solution at all.")
    return best_selected_incidence_matrix, lowest_stencil_variance, best_co_occurrence_matrix, super_pools


def get_co_occurrence_matrix(A:jnp.ndarray):
    return A @ A.T


if __name__ == "__main__":
    A, variance, C, superpool = get_best_unordered_races(P, N)
    print(A)
    print(A.sum(0))
    print(C)
    print(f"{variance=}")