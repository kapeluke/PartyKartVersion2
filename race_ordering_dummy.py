import random

def generate_schedule(p_count: int, n_races: int, player_ids: list) -> list:
    """
    MOCK ALGORITHM: Replace this with your actual mathematical logic.
    Returns a list of dictionaries representing the races.
    Every race will have at minimum 3 players and at most 4.
    """
    races = []
    
    # Just a dummy generation to populate the UI for testing
    # It creates enough races so everyone races N times (roughly)
    total_slots = p_count * n_races
    num_races = total_slots // 4
    
    if total_slots % 4 != 0:
        num_races += 1

    # Flatten a list of players N times and shuffle
    pool = player_ids * n_races
    random.shuffle(pool)

    race_id = 1
    while pool:
        # Take 4 players for a race (or 3 if running out)
        chunk_size = 4 if len(pool) >= 4 else 3
        if len(pool) < 3:
             chunk_size = len(pool) # edge case for end of pool in mock
             
        current_players = pool[:chunk_size]
        pool = pool[chunk_size:]
        
        # Ensure no duplicates in a single race in this mock
        # (Your real algorithm will handle this perfectly)
        current_players = list(set(current_players)) 
        
        # Pad with randoms if set() shrunk it (just for the mock not to crash)
        while len(current_players) < 3 and player_ids:
            p = random.choice(player_ids)
            if p not in current_players:
                current_players.append(p)

        races.append({
            "id": race_id,
            "players": current_players
        })
        race_id += 1

    return races