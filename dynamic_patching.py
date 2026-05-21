import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix
from itertools import combinations
import race_ordering
import race_selection

""" This file's intention is to be similar to race_selection and race_ordering BUT with one key difference:
the methods in here deal with the case of the total players (P) changing after a certain number of races have already completed.

So most of the logic and solving is similar, but with enough key differences that if it were in the other files it would make them too messy.

I refer to $F$ here as the number of finished races. That is, the race at index $F$ is the "current" race that we can start overwriting.
All races of indices $[0..F-1]$ must be kept constant in ordering and their rosters.
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
    """Returns the number of races needed for every player in the incidence matrix `A` 
    after `F` races have already completed. It should be aware of dropped players.
    """
    # FIX: Use each player's individual lifetime target currently recorded in the schedule
    individual_lifetime_targets = A.sum(1)
    completed_races = A[:,:F].sum(1)
    return individual_lifetime_targets - completed_races


# -------------- GHOST RACES THRESHOLD HANDLING --------------
def _ghost_race_add_safe_boundaries(core_targets:jnp.ndarray) -> jnp.ndarray:
    S_core = core_targets.sum()
    N_max_core = core_targets.max()
    safe_max = jnp.floor(S_core / 2)
    safe_min = max(1, 3 * N_max_core - S_core)
    return safe_min, safe_max


def get_ghost_safe_boundaries(A:jnp.ndarray, F:int) -> tuple[int,int]:
    safe_min, safe_max = _ghost_race_add_safe_boundaries(_core_targets(A, F))
    return int(safe_min), int(safe_max)


# -------------- SPECIES GENERATION (CANDIDATES) --------------
def _generate_patch_candidates(P_old:int, G:int, stencil:jnp.ndarray) -> jnp.ndarray:
    """Generates the candidate pool of size 3 and 4 races, including swapped Gary species."""
    P_total = P_old + G
    shifts = np.concatenate([np.array([0]), np.array(stencil)])
    base_holes = np.cumsum(shifts) % P_old
    rotations = np.arange(P_old)[:, None]
    size_4_holes = (base_holes + rotations) % P_old 
    
    all_cols = []
    # 1. Type A: Pure Core 4
    for i in range(P_old):
        col = np.zeros(P_total, dtype=np.int32)
        col[size_4_holes[i]] = 1
        all_cols.append(col)
    
    # Combinations of 3 (Core holes)
    for combo in combinations(range(4), 3):
        for i in range(P_old):
            holes_3 = size_4_holes[i, list(combo)]
            # Type B: 3 Core + Ghost (Size 3)
            col = np.zeros(P_total, dtype=np.int32)
            col[holes_3] = 1
            all_cols.append(col)
            # Type C: 3 Core + 1 Gary (Size 4)
            for g in range(G):
                col_g = col.copy()
                col_g[P_old + g] = 1
                all_cols.append(col_g)
                
    # Combinations of 2 (Core holes)
    for combo in combinations(range(4), 2):
        for i in range(P_old):
            holes_2 = size_4_holes[i, list(combo)]
            # Awkward 2 Core + Ghost (Size 2 - Highly penalized)
            col = np.zeros(P_total, dtype=np.int32)
            col[holes_2] = 1
            all_cols.append(col)
            # Type E: 2 Core + 1 Gary + Ghost (Size 3)
            for g in range(G):
                col_g = col.copy()
                col_g[P_old + g] = 1
                all_cols.append(col_g)
            # Type D: 2 Core + 2 Gary (Size 4)
            for g1, g2 in combinations(range(G), 2):
                col_gg = col.copy()
                col_gg[P_old + g1] = 1
                col_gg[P_old + g2] = 1
                all_cols.append(col_gg)
                
    return jnp.array(all_cols).T

def _generate_gary_centric_candidates(P_old:int, G:int) -> jnp.ndarray:
    """Generates races consisting exclusively of Garys (needed if G is large)."""
    P_total = P_old + G
    all_cols = []
    for size in [1, 2, 3, 4]:
        if size > G: continue
        for combo in combinations(range(G), size):
            col = np.zeros(P_total, dtype=np.int32)
            for g in combo:
                col[P_old + g] = 1
            all_cols.append(col)
    if all_cols:
        return jnp.array(all_cols).T
    return jnp.zeros((P_total, 0), dtype=jnp.int32)


# -------------- MILP SOLVER & SCORE UTILS --------------
def _construct_column_cost_vector(candidates:jnp.ndarray) -> jnp.ndarray:
    """Take the array of incidence's/candidate races and write out the cost of choosing them in a vector."""
    costs = jnp.ones((candidates.shape[1],))
    for size, cost in RACE_COSTS_BY_SIZE.items():
        costs = jnp.where(candidates.sum(0) == size, cost, costs)
    return costs


def _calculate_dynamic_min_overlap(lower_bounds, upper_bounds) -> int:
    """Dynamically determines the absolute mathematical floor for max_overlap
    to prevent the ILP solver from wasting time on impossible bounds.
    """
    # Convert from JAX arrays to NumPy for safe condition checks
    lb = np.array(lower_bounds)
    ub = np.array(upper_bounds)
    active_mask = ub > 0
    N_active = int(np.sum(active_mask))
    # If there are 1 or fewer active players, overlap limits are trivial
    if N_active <= 1:
        return 1
    # BOUND 1: Individual player pigeonhole principle
    # Assuming a conservative minimum race size of 3, each race gives a player 
    # at least 2 pairings. These must be spread across the other (N_active - 1) players.
    active_lowers = lb[active_mask]
    pigeonhole_bound = int(np.max(np.ceil(2 * active_lowers / (N_active - 1))))
    # BOUND 2: Global pair density floor
    # Assuming a minimum race size of 3, total pairs generated >= total slots filled.
    total_slots = int(np.sum(active_lowers))
    unique_pairs = (N_active * (N_active - 1)) // 2
    global_density_bound = int(np.ceil(total_slots / unique_pairs))
    return max(1, pigeonhole_bound, global_density_bound)


def _get_masked_variance(A_sel:jnp.ndarray, active_indices:jnp.ndarray) -> float:
    """Calculates pairwise variance while strictly ignoring dropped players."""
    C = A_sel @ A_sel.T
    C_active = C[jnp.ix_(active_indices, active_indices)]
    mask = ~jnp.eye(C_active.shape[0], dtype=bool)
    return C_active[mask].var()


def _solve_dynamic_ilp(A_super:jnp.ndarray, lower_bounds:jnp.ndarray, upper_bounds:jnp.ndarray, costs:jnp.ndarray, max_overlap:int, ilp_timeout:float, shuffle_seed:int=42) -> jnp.ndarray:
    num_candidates = A_super.shape[1]
    P_total = A_super.shape[0]
    key = jax.random.key(shuffle_seed)
    shuffle_idx = jax.random.permutation(key, num_candidates)
    shuffled_incidence = A_super[:, shuffle_idx]
    shuffled_costs = costs[shuffle_idx]
    i_idx, j_idx = jnp.triu_indices(P_total, k=1)
    pairwise_overlaps = shuffled_incidence[i_idx] * shuffled_incidence[j_idx]
    A_constrained = jnp.vstack([shuffled_incidence, pairwise_overlaps])
    lb = jnp.concatenate([lower_bounds, jnp.zeros(len(i_idx))])
    ub = jnp.concatenate([upper_bounds, jnp.full(len(i_idx), max_overlap)])
    A_sparse = csc_matrix(np.array(A_constrained))
    constraints = LinearConstraint(A_sparse, lb=lb, ub=ub)
    # FIX 1: Allow columns to be selected multiple times by setting bound ceiling to max remaining races
    max_races = int(upper_bounds.max())
    bounds = Bounds(np.zeros(num_candidates), np.full(num_candidates, max_races))
    integrality = np.ones(num_candidates)
    
    options = {'time_limit': ilp_timeout} 
    result = milp(c=np.array(shuffled_costs), constraints=constraints, integrality=integrality, bounds=bounds, options=options)
    
    # FIX 2: Correctly duplicate columns when counts are greater than 1 using np.repeat
    if result.success and result.x is not None:
        chosen_counts = np.round(result.x).astype(int)
        selected_shuffled_indices = np.repeat(np.arange(num_candidates), chosen_counts)
        if len(selected_shuffled_indices) == 0:
            return None
        selected_original_indices = shuffle_idx[selected_shuffled_indices]
        return A_super[:, selected_original_indices]
    return None


def _solve_dynamic_ilp_auto_relaxing(A_super, lower_bounds, upper_bounds, costs, A_past, active_indices, ilp_timeout=10.0, shuffle_seed=42):
    current_max_overlap = 1
    absolute_ceiling = 6 
    best_A_sel = None
    best_variance = float('inf')
    best_cost = float('inf')
    while current_max_overlap <= absolute_ceiling:
        A_sel = _solve_dynamic_ilp(A_super, lower_bounds, upper_bounds, costs, current_max_overlap, ilp_timeout, shuffle_seed=shuffle_seed)
        if A_sel is not None:
            # Construct the full prospective schedule to see true global tournament variance
            A_full_temp = jnp.concatenate([A_past, A_sel], axis=1)
            # FIX: Calculate variance *strictly* masking out dropped/inactive players
            current_variance = _get_masked_variance(A_full_temp, active_indices)
            # Calculate tournament length/cost (fewer races = lower cost)
            current_cost = float(_construct_column_cost_vector(A_sel).sum())
            # Priority 1: Keep variance among active players to an absolute minimum
            if current_variance < best_variance:
                best_variance = current_variance
                best_cost = current_cost
                best_A_sel = A_sel
            # Priority 2: Tie-breaker prefers shorter tournament lengths (lower cost)
            elif np.isclose(current_variance, best_variance) and current_cost < best_cost:
                best_variance = current_variance
                best_cost = current_cost
                best_A_sel = A_sel
        current_max_overlap += 1
    return best_A_sel

# -------------- ORDER THE NEW RACES --------------
def _sa_patch_ordering(A_past:jnp.ndarray, A_patch:jnp.ndarray, base_climbers:int=5000, base_steps:int=1500, seed:int=42) -> jnp.ndarray:
    """Permutes only A_patch, leaving A_past rigidly locked in time, dynamically scaling hyperparameters."""
    rng = jax.random.key(seed)
    R_patch = A_patch.shape[1]
    if R_patch <= 1:
        return jnp.arange(R_patch)
        
    R_total = A_past.shape[1] + R_patch
    ratio = R_patch / R_total
    
    num_climbers = max(500, int(base_climbers * ratio))
    steps = max(200, int(base_steps * ratio))
    
    rng, rng_loop = jax.random.split(rng)
    keys = jax.random.split(rng, num_climbers)
    initial_pis = jax.vmap(jax.random.permutation, in_axes=(0,None))(keys, R_patch)
    
    @jax.jit
    def score_patch(pi, A_past, A_patch):
        A_full = jnp.concatenate([A_past, A_patch[:, pi]], axis=1)
        R = A_full.shape[1]
        indices = jnp.arange(R)
        masked = jnp.where(A_full, indices, 999)
        player_races = jnp.sort(masked, axis=1)
        gaps = jnp.diff(player_races, axis=1)
        valid_gaps = gaps < R
        
        counts = A_full.sum(1)
        mu = jnp.where(counts > 1, (R - 1) / (counts - 1), 0.0)
        sq_diffs = jnp.square(gaps - mu[:, None]) * valid_gaps
        return jnp.sum(sq_diffs)

    vmap_score = jax.vmap(score_patch, in_axes=(0, None, None))
    vmap_mutate = jax.vmap(race_ordering.mutate_permutation, in_axes=(0, 0))
    
    initial_scores = vmap_score(initial_pis, A_past, A_patch)
    anneal_steps = int(steps * 0.8)
    greedy_steps = steps - anneal_steps
    temps = jnp.concatenate([jnp.linspace(10.0, 0.0, anneal_steps), jnp.zeros(greedy_steps)])
    
    @jax.jit
    def sa_step(carry, step_data):
        current_pis, current_scores = carry
        step_key, temp = step_data
        mutate_keys = jax.random.split(step_key, num_climbers)
        proposed_pis = vmap_mutate(current_pis, mutate_keys)
        proposed_scores = vmap_score(proposed_pis, A_past, A_patch)
        delta_E = proposed_scores - current_scores
        probs = jnp.where(temp > 0, jnp.exp(-delta_E / (temp + 1e-9)), 0.0)
        rand_vals = jax.random.uniform(step_key, shape=(num_climbers,))
        accept = (delta_E < 0) | (rand_vals < probs)
        next_pis = jnp.where(accept[:,None], proposed_pis, current_pis)
        next_scores = jnp.where(accept, proposed_scores, current_scores)
        return (next_pis, next_scores), None
        
    step_keys = jax.random.split(rng_loop, steps)
    (final_pis, final_scores), _ = jax.lax.scan(sa_step, (initial_pis, initial_scores), (step_keys, temps))
    best_idx = jnp.argmin(final_scores)
    return final_pis[best_idx]


# -------------- ADD AND DROP PLAYER METHODS --------------
def drop_players(super_pools:jnp.ndarray, A:jnp.ndarray, F:int, dropped_indices:list[int], **kwargs) -> jnp.ndarray:
    """Handle the dropping of players by setting their future targets to 0."""
    seed = kwargs.get('seed', 42)
    ilp_timeout = kwargs.get('ilp_timeout', 10.0)
    num_restarts = kwargs.get('num_restarts', 3)
    pool_size = kwargs.get('pool_size', super_pools.shape[1])
    num_climbers = kwargs.get('num_climbers', 5000)
    num_steps = kwargs.get('num_steps', 1500)

    P_old = A.shape[0]
    core_targets = np.array(_core_targets(A, F))
    
    for idx in dropped_indices:
        core_targets[idx] = 0
    core_targets = jnp.array(core_targets)
    
    lower_bounds = core_targets
    upper_bounds = core_targets
    actual_pool_size = min(pool_size, super_pools.shape[1])
    
    best_A_patch = None
    lowest_var = jnp.inf
    
    # Strictly typing active indices to a JAX array
    active_indices = jnp.array([i for i in range(P_old) if i not in dropped_indices])
    
    # FIX 3: Generate active-player-only fallback combinations to prevent stencil starvation
    active_cols = []
    if len(active_indices) <= 25:
        raw_active_list = [int(i) for i in active_indices]
        for combo in combinations(raw_active_list, 4):
            col = np.zeros(P_old, dtype=np.int32)
            col[list(combo)] = 1
            active_cols.append(col)
        for combo in combinations(raw_active_list, 3):
            col = np.zeros(P_old, dtype=np.int32)
            col[list(combo)] = 1
            active_cols.append(col)
            
    if active_cols:
        A_active_combos = jnp.array(active_cols).T
    else:
        A_active_combos = jnp.zeros((P_old, 0), dtype=jnp.int32)
    
    A_past = A[:, :F]

    for chunk_idx in range(super_pools.shape[0]):
        chunk = super_pools[chunk_idx]
        matrices = []
        for i in range(actual_pool_size):
            A_part, _ = race_selection.generate_batch(P_old, chunk[i], race_selection.CUTS_PER_ROTATION)
            matrices.append(A_part)
            
        # Append flexible combinations to the candidate pool
        matrices.append(A_active_combos)
            
        A_super = jnp.concatenate(matrices, axis=1)
        A_super = jnp.clip(A_super, 0, 1) # Defensive clip mapping back to pure binary state
        costs = _construct_column_cost_vector(A_super)    
        
        for restart in range(num_restarts):
            # Pass A_past and active_indices to ensure correct inner-loop sorting
            A_patch = _solve_dynamic_ilp_auto_relaxing(
                A_super, lower_bounds, upper_bounds, costs, 
                A_past=A_past, active_indices=active_indices, 
                ilp_timeout=ilp_timeout, shuffle_seed=seed+restart
            )
            if A_patch is not None:
                A_full_temp = jnp.concatenate([A_past, A_patch], axis=1)
                var = _get_masked_variance(A_full_temp, active_indices)
                if var < lowest_var:
                    lowest_var = var
                    best_A_patch = A_patch
                    
    if best_A_patch is None:
        print("Failed to find a valid schedule to drop players.")
        return A
        
    pi = _sa_patch_ordering(A[:, :F], best_A_patch, base_climbers=num_climbers, base_steps=num_steps, seed=seed)
    A_patch_ordered = best_A_patch[:, pi]
    
    return jnp.concatenate([A[:, :F], A_patch_ordered], axis=1)


def add_players(super_pools:jnp.ndarray, A:jnp.ndarray, F:int, boundaries:list[tuple[int,int]], **kwargs) -> jnp.ndarray:
    """Handle the addition of new players to the roster after F races using species swaps."""
    seed = kwargs.get('seed', 42)
    ilp_timeout = kwargs.get('ilp_timeout', 10.0)
    num_restarts = kwargs.get('num_restarts', 3)
    pool_size = kwargs.get('pool_size', super_pools.shape[1])
    num_climbers = kwargs.get('num_climbers', 5000)
    num_steps = kwargs.get('num_steps', 1500)

    P_old = A.shape[0]
    G = len(boundaries)
    P_total = P_old + G
    
    races_needed_for_core_racers = _core_targets(A, F)
    boundaries_matrix = jnp.array(boundaries)
    lower_bounds = jnp.concatenate([races_needed_for_core_racers, boundaries_matrix[:,0]])
    upper_bounds = jnp.concatenate([races_needed_for_core_racers, boundaries_matrix[:,1]])
    
    A_padded = jnp.vstack([A, jnp.zeros((G, A.shape[1]), dtype=jnp.int32)])
    A_past = A_padded[:, :F]
    
    actual_pool_size = min(pool_size, super_pools.shape[1])
    best_A_patch = None
    lowest_var = jnp.inf
    active_indices = jnp.arange(P_total)
    
    for chunk_idx in range(super_pools.shape[0]):
        chunk = super_pools[chunk_idx]
        matrices = []
        for i in range(actual_pool_size):
            A_part = _generate_patch_candidates(P_old, G, chunk[i])
            matrices.append(A_part)
            
        matrices.append(_generate_gary_centric_candidates(P_old, G))
        
        A_super = jnp.concatenate(matrices, axis=1)
        A_super = jnp.clip(A_super, 0, 1)
        costs = _construct_column_cost_vector(A_super)
        
        for restart in range(num_restarts):
            A_patch = _solve_dynamic_ilp_auto_relaxing(
                A_super, lower_bounds, upper_bounds, costs, 
                A_past=A_past, active_indices=active_indices, 
                ilp_timeout=ilp_timeout, shuffle_seed=seed+restart
            )
            if A_patch is not None:
                A_full_temp = jnp.concatenate([A_past, A_patch], axis=1)
                var = _get_masked_variance(A_full_temp, active_indices)
                if var < lowest_var:
                    lowest_var = var
                    best_A_patch = A_patch 
                    
    if best_A_patch is None:
        print("Failed to find a valid schedule to add players.")
        return A_padded
        
    pi = _sa_patch_ordering(A_past, best_A_patch, base_climbers=num_climbers, base_steps=num_steps, seed=seed)
    A_patch_ordered = best_A_patch[:, pi]
    
    return jnp.concatenate([A_past, A_patch_ordered], axis=1)


if __name__ == "__main__":
    # Testing the severe drop scenario
    A = jnp.array([
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    ])
    super_pools = jnp.array([[[1, 1, 1]]])
    
    print("Testing Drop Players at F=0:")
    A_new_f0 = drop_players(super_pools, A, F=0, dropped_indices=[4, 5, 6, 7, 8], num_restarts=1)
    print(A_new_f0)
    print("Patched R at F=0:", A_new_f0.shape[1])
    
    print("\nTesting Drop Players at F=1:")
    A_new_f1 = drop_players(super_pools, A, F=1, dropped_indices=[4, 5, 6, 7, 8], num_restarts=1)
    print(A_new_f1)
    print("Patched R at F=1:", A_new_f1.shape[1])