import jax
import jax.numpy as jnp
import numpy as np
from ortools.sat.python import cp_model


"""This file's goal solves the SECOND problem:

How do we order a set of races with 3 or 4 players in each of them, such that every players' races are spread out as evenly as possible.

So this does not mean that every player races their `N` races all in a row and then rests the entire night. That would technically be _fair_
but would be super boring for them. So we want their races to be spread out. We can measure the error/loss as some sum of squared errors,
where each individual error is the distance of a players' races from the ideal, $\\mu=\\frac{R-1}{n_{races}-1}$

There are two ways to solve this:

1. Solve it similarly to how we solved the "selection" problem with an ILP/MILP solver. This produces some great solutions and are much
better mathematically than option 2. The problem is the solver takes a _very_ long time for any schedule of races longer than something like
15. So we decide to only use it when the number of races is under 15.

2. Solve it with Simulated Annealing (SA). We start with an ordering of the set of races, and then perform swaps if they improve our score.
This is prone to getting stuck in local optima, so with some random probability ($\\exp(-\\delta E / T)$) we take a suboptimal move. Where
$\\delta E$ is the difference in energy (score) and $T$ is the temperature. We start in a higher entropy environment (high $T$) and slowly
lower temperature as time goes on to start taking more and more optimal moves. For the final 20% of execution, we set $T=0$ to mean we only
take the optimal moves. We do this for some number of parallel "climbers" over some number of steps and return the best solution.
"""


SA_NUM_CLIMBERS = 10000
SA_STEPS = 3000
EPSILON = 1e-9
MILP_TIMEOUT = 60.0 # seconds


def score_target_gaps(pi:jnp.ndarray, A_sel:jnp.ndarray, N:int) -> float:
    """Calculate exactly how far the gaps deviates from the ideal, \\mu."""
    R = A_sel.shape[1]
    mu = (R - 1) / (N - 1)
    # Reorder tournament according to pi
    A_perm = A_sel[:,pi]
    # Extract race indices for each player
    indices = jnp.arange(R)
    masked = jnp.where(A_perm, indices, 999)
    player_races = jnp.sort(masked, axis=1)[:,:N]
    # Calculate exact internal gaps between races
    gaps = jnp.diff(player_races, axis=1)
    # Return squared distance from perfect target gap
    return jnp.sum(jnp.square(gaps - mu))


def mutate_permutation(pi:jnp.ndarray, rng:jnp.ndarray) -> jnp.ndarray:
    """Randomly swap two races."""
    idx1, idx2 = jax.random.choice(rng, pi.shape[0], shape=(2,), replace=False)
    return pi.at[idx1].set(pi[idx2]).at[idx2].set(pi[idx1])


def simulated_annealing(rng:jnp.ndarray, A_sel:jnp.ndarray, N:int, num_climbers:int=SA_NUM_CLIMBERS, steps:int=SA_STEPS):
    R = A_sel.shape[1]
    rng, rng_loop = jax.random.split(rng)
    # Initialize random starting point
    keys = jax.random.split(rng, num_climbers)
    initial_pis = jax.vmap(jax.random.permutation, in_axes=(0,None))(keys, R)
    vmap_score = jax.vmap(score_target_gaps, in_axes=(0,None,None))
    vmap_mutate = jax.vmap(mutate_permutation, in_axes=(0, 0))
    initial_scores = vmap_score(initial_pis, A_sel, N)
    # Create cooling schedule: High heat down to 0 then stay at 0 for pure greedy
    anneal_steps = int(steps * 0.8) # 80% of time spent with >0 heat. Decrease in this number of steps
    greedy_steps = steps - anneal_steps # pure greedy last 20% of steps
    # Start the temperature based roughly on expected delta E
    temps = jnp.concatenate([
        jnp.linspace(10.0, 0.0, anneal_steps),
        jnp.zeros(greedy_steps)
    ])
    # Define the loop
    @jax.jit
    def sa_step(carry, step_data):
        current_pis, current_scores = carry
        step_key, temp = step_data
        # 1. Propose mutated schedules
        mutate_keys = jax.random.split(step_key, num_climbers)
        proposed_pis = vmap_mutate(current_pis, mutate_keys)
        proposed_scores = vmap_score(proposed_pis, A_sel, N)
        # 2. Calculate energy delta
        delta_E = proposed_scores - current_scores
        # 3. Acceptance probability: exp(-\delta E / T)
        # If temp is 0, probability is 0 (unless delta_E is negative)
        probs = jnp.where(temp > 0, jnp.exp(-delta_E / (temp + EPSILON)), 0.0)
        # 4. Roll dice
        rand_vals = jax.random.uniform(step_key, shape=(num_climbers,))
        # Accept choice if there's a beter move or if the dice roll went well
        accept = (delta_E < 0) | (rand_vals < probs)
        # 5. Update states
        next_pis = jnp.where(accept[:,None], proposed_pis, current_pis)
        next_scores = jnp.where(accept, proposed_scores, current_scores)
        return (next_pis, next_scores), None
    # Run the loop
    step_keys = jax.random.split(rng_loop, steps)
    step_data = (step_keys, temps)
    (final_pis, final_scores), _ = jax.lax.scan(sa_step, (initial_pis, initial_scores), step_data)
    best_idx = jnp.argmin(final_scores)
    return final_pis[best_idx]


def solve_milp(A_sel:np.ndarray) -> jnp.ndarray:
    """
    Finds the guaranteed mathematical global minimum schedule using the OR-Tools CP-SAT solver.
    """
    P, R = A_sel.shape
    model = cp_model.CpModel()
    # 1. THE PERMUTATION VARIABLES
    # race_slots[r] represents the scheduled time slot (0 to R-1) for race r.
    race_slots = [model.NewIntVar(0, R - 1, f'race_slot_{r}') for r in range(R)]
    # This single line replaces hundreds of matrix constraints in SciPy
    model.AddAllDifferent(race_slots)
    gap_penalties = []
    # 2. EXTRACTING GAPS PER PLAYER
    for p in range(P):
        player_races = np.where(A_sel[p])[0]
        N_p = len(player_races)
        if N_p <= 1:
            continue # No gaps to measure for this player
        # We need integer math for the solver. We scale mu by 10 to preserve a decimal.
        mu_scaled = int(((R - 1) / (N_p - 1)) * 10)
        # S will hold the chronologically sorted time slots for player p
        S = [model.NewIntVar(0, R - 1, f'S_{p}_{i}') for i in range(N_p)]
        # Force S to be strictly increasing (this simulates sorting)
        for i in range(N_p - 1):
            model.Add(S[i] < S[i+1])
        # Match the player's unsorted race slots to the sorted S array
        # We create a boolean matrix to link them
        matches = {}
        for i in range(N_p):
            for j in range(N_p):
                matches[(i, j)] = model.NewBoolVar(f'match_{p}_{i}_{j}')
        for i in range(N_p):
            model.AddExactlyOne([matches[(i, j)] for j in range(N_p)])
        for j in range(N_p):
            model.AddExactlyOne([matches[(i, j)] for i in range(N_p)])
        for i in range(N_p):
            for j, r in enumerate(player_races):
                # If match is true, S[i] gets the time slot of race r
                model.Add(S[i] == race_slots[r]).OnlyEnforceIf(matches[(i, j)])
        # 3. CALCULATE THE EXACT OBJECTIVE
        for i in range(N_p - 1):
            gap = model.NewIntVar(1, R, f'gap_{p}_{i}')
            model.Add(gap == S[i+1] - S[i])
            gap_scaled = model.NewIntVar(10, R * 10, f'gap_scaled_{p}_{i}')
            model.Add(gap_scaled == gap * 10)
            # difference = actual_gap - ideal_mu
            diff = model.NewIntVar(-R * 10, R * 10, f'diff_{p}_{i}')
            model.Add(diff == gap_scaled - mu_scaled)
            # square the difference
            sq_diff = model.NewIntVar(0, (R * 10)**2, f'sq_diff_{p}_{i}')
            model.AddMultiplicationEquality(sq_diff, [diff, diff])
            gap_penalties.append(sq_diff)
    # 4. MINIMIZE THE TOTAL VARIANCE
    model.Minimize(sum(gap_penalties))
    # 5. SOLVE
    solver = cp_model.CpSolver()
    # Add a time limit just in case it gets bogged down on R > 15
    solver.parameters.max_time_in_seconds = MILP_TIMEOUT 
    print("Solving exact formulation...")
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if status == cp_model.OPTIMAL:
            print("Mathematical Global Minimum PROVEN.")
        else:
            print("Sub-optimal but feasible solution found (hit time limit).")
        # Reconstruct the permutation sequence (pi)
        pi = [0] * R
        for r in range(R):
            slot = solver.Value(race_slots[r])
            pi[slot] = r
        return jnp.array(pi)
    else:
        print("No solution found.")
        return None


def find_best_race_order(seed:int, A_sel:jnp.ndarray, N:int, num_climbers:int=SA_NUM_CLIMBERS, steps:int=SA_STEPS):
    """Dynamically routes to the best algorithm based on matrix dimensions.
    We use simulated annealing for R > 15 but MILP for R <= 15.
    MILP gives proven better solutions but becomes infeasible for R > 15.
    """
    rng = jax.random.key(seed)
    R = A_sel.shape[1]
    if R <= 15:
        print(f"{R=} detected. Search space is small enough. Routing to MILP solver ...")
        return solve_milp(A_sel)
    else:
        print(f"{R=} detected. Search space is too large. Routing to vectorized simulated annealing ...")
        return simulated_annealing(rng, A_sel, N, num_climbers, steps)


def permute_incidence_matrix(A:jnp.ndarray, permutation_order):
    return A[:,permutation_order]


def get_indices_of_ones(A:jnp.ndarray):
    return jnp.argwhere(A == 1)


if __name__ == "__main__":
    import race_selection
    P = 17
    N = 5
    SEED = 42
    A, variance, C, superpool = race_selection.get_best_unordered_races(P, N, strict_R=False, strictness_tolerance=1, cuts_per_rotation=3, num_stencils=20, pool_size=3, restarts=10, seed=SEED)
    R = A.shape[1]
    print(A)
    print(A.sum(0))
    print(race_selection.get_co_occurrence_matrix(A))
    permutation_order = find_best_race_order(SEED, A, 1000, 2000)
    print(permutation_order)
    permuted_A = permute_incidence_matrix(A, permutation_order)
    print(permuted_A)

    ones_indices = get_indices_of_ones(permuted_A)
    player_races = {p:[] for p in range(P)}
    for (p,r) in ones_indices:
        player_races[int(p)].append(int(r))
    print(player_races)

    roster_in_each_race = [
        {
            'id': r + 1,
            'players': []
        } for r in range(permuted_A.shape[1])]
    for (p,r) in ones_indices:
        roster_in_each_race[int(r)]['players'].append(int(p))
    print(roster_in_each_race)
    # print({p:int(jnp.diff(jnp.array([0] + r + [R])).sum()) for p,r in player_races.items()})