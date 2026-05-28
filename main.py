from fastapi import FastAPI, Request, Response, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
from dataclasses import dataclass
from typing import Optional
from jax.numpy import ndarray

# The race schedule-making algorithm
import race_selection
import race_ordering
import modifier_shuffling
import dynamic_patching


app = FastAPI()
templates = Jinja2Templates(directory="templates")


HYPERPARAMETER_DEFAULTS = {
    'strict_R': race_selection.STRICT_R,
    'strictness_tolerance': 2,
    'cuts_per_rotation': race_selection.CUTS_PER_ROTATION,
    'num_stencils': 20,
    'pool_size': race_selection.SUPERPOOL_SIZE,
    'num_restarts': race_selection.NUM_RESTARTS,
    'seed': 42,
    'gap_variance_weight': race_selection.GAP_VARIANCE_PENALTY_WEIGHT,
    'num_climbers': race_ordering.SA_NUM_CLIMBERS,
    'num_steps': race_ordering.SA_STEPS,
    'ilp_timeout': race_selection.ILP_MAX_TIME_LIMIT,
    'milp_timeout': race_ordering.MILP_TIMEOUT,
}


# --- IN-MEMORY STATE ---
# For a production app, you'd use a database, but for a single-night tournament 
#  hosted locally, a global memory state is perfect and incredibly fast.
class TournamentState(BaseModel):
    is_active: bool = False
    p_count: int = 5
    n_races: int = 4
    players: dict = {}  # {player_id: {"name": str, "points": int, "active": bool}}
    races: list = []    # [{"id": int, "players": [id1, id2, ...], "modifiers": [{"short_text": str, "description": str}, ...], "results": {}}, ...]
    current_race_idx: int = 0
    history: list = []  # Stack for undo functionality
    hyperparameters: dict = HYPERPARAMETER_DEFAULTS.copy()
    modifiers: list = [] # [{"channel": int, "short_text": str, "description": str}, ...]

app_state = TournamentState()


@dataclass
class BackendCache:
    incidence_matrix:Optional[ndarray] = None
    super_pools:Optional[ndarray] = None
    shuffled_modifiers:Optional[list[dict[str,str]]] = None

backend_cache = BackendCache()


# Points map (Mario Kart style points can be adjusted here)
POINTS_SYSTEM = {
    4: {1: 24, 2: 16, 3: 8, 4: 0},
    3: {1: 20, 2: 10, 3: 0}
}


# --- DEPENDENCIES & HELPERS ---
def get_sorted_leaderboard():
    global app_state
    players_list = list(app_state.players.values())
    return sorted(players_list, key=lambda x: x["points"], reverse=True)


def save_history():
    """Deep copy the current state (minus history) to the history stack."""
    global app_state
    state_copy = app_state.model_dump()
    state_copy["history"] = [] # Don't nest history inside history
    app_state.history.append(state_copy)


# --- ENDPOINTS ---
@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    global app_state
    """Renders the main page. Jinja handles whether to show Setup or Tournament."""
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "state": app_state,
            "leaderboard": get_sorted_leaderboard() if app_state.is_active else [],
            # **app_state.hyperparameters
        }
    )


@app.get('/components/player_inputs', response_class=HTMLResponse)
async def get_player_inputs(request: Request, p_count: int = 5):
    """HTMX endpoint to dynamically generate name inputs when the P slider changes."""
    global app_state
    # Enforce constraints
    p_count = max(5, min(25, p_count))
    html = ""
    for i in range(p_count):
        html += f'''
        <div class="flex flex-col relative">
            <span class="absolute -top-2 -left-2 bg-yellow-400 text-black font-bold border-2 border-black rounded-full w-6 h-6 flex items-center justify-center text-xs z-10">{i+1}</span>
            <input type="text" name="player_names" placeholder="Name" value="{app_state.players.get(i,{}).get('name','')}" required
                    class="bg-white text-gray-900 border-2 border-black rounded-xl p-3 font-bold focus:ring-4 focus:ring-yellow-400 outline-none shadow-[2px_2px_0_#000]">
        </div>
        '''
    return html


@app.post('/generate', response_class=HTMLResponse)
async def generate_tournament(
    request: Request, 
    p_count: int = Form(...), 
    n_races: int = Form(...),
    player_names: list[str] = Form(...)):
    """Handles the form submission, triggers the algorithm, and starts the tournament."""
    # await asyncio.sleep(2) 

    # Reset state
    global app_state
    app_state.is_active = True
    app_state.p_count = p_count
    app_state.n_races = n_races
    app_state.players = {i: {"id": i, "name": name, "points": 0, "active": True} for i, name in enumerate(player_names)}
    app_state.current_race_idx = 0

    # app_state.hyperparameters['ilp_timeout'] = float(ilp_timeout) # unused at the moment
    # app_state.hyperparameters['milp_timeout'] = float(milp_timeout) # unused at the moment

    # app_state.races = race_selection.generate_schedule(p_count, n_races, list(app_state.players.keys()))
    kwargs_selection = {
        'strict_R':app_state.hyperparameters['strict_R'],
        'strictness_tolerance':app_state.hyperparameters['strictness_tolerance'],
        'cuts_per_rotation':app_state.hyperparameters['cuts_per_rotation'],
        'num_stencils':app_state.hyperparameters['num_stencils'],
        'pool_size':app_state.hyperparameters['pool_size'],
        'restarts':app_state.hyperparameters['num_restarts'],
        'seed':app_state.hyperparameters['seed'],
        'gap_variance_weight':app_state.hyperparameters['gap_variance_weight']
    }

    A, variance, C, super_pools = race_selection.get_best_unordered_races(p_count, n_races, **kwargs_selection)
    print(A)
    print(C)

    kwargs_ordering = {
        'steps':app_state.hyperparameters['num_steps'],
        'num_climbers':app_state.hyperparameters['num_climbers'],
        # 'seed':app_state.hyperparameters['seed'],
    }

    permutation_order = race_ordering.find_best_race_order(app_state.hyperparameters['seed'], A, n_races, **kwargs_ordering)
    permuted_A = race_ordering.permute_incidence_matrix(A, permutation_order)
    print(permuted_A)

    backend_cache.incidence_matrix = permuted_A
    backend_cache.super_pools = super_pools # safe the super pools so we can reuse them later for adding and dropping players

    ones_indices = race_ordering.get_indices_of_ones(permuted_A)
    roster_in_each_race = [{
        'id': int(r),
        'players': [],
    } for r in range(permuted_A.shape[1])]
    for (p,r) in ones_indices:
        race_id = int(r)
        racer_id = int(p)
        roster_in_each_race[race_id]['players'].append(racer_id)
    print(roster_in_each_race)

    app_state.races = roster_in_each_race

    # Shuffle the modifiers and apply to each race
    num_modifiers_to_get = len(app_state.races)
    modifiers = modifier_shuffling.shuffle_out_modifiers(app_state.hyperparameters['seed'], num_modifiers_to_get, app_state.modifiers)
    backend_cache.shuffled_modifiers = modifiers # store them for later for adding and dropping players
    for idx in range(len(app_state.races)):
        app_state.races[idx]['modifiers'] = modifiers[idx]

    # Return the full page (HTMX will swap the whole body)
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"state": app_state, "leaderboard": get_sorted_leaderboard()}
    )


@app.post('/api/save-hyperparameters')
async def save_hyperparameters(request: Request, response: Response):
    """HTMX endpoint to dynamically save all algorithm hyperparameters from the modal."""
    form_data = await request.form()
    global app_state
    # Dynamically update the hyperparameters based on what was submitted
    print(form_data.items())
    for key, current_value in app_state.hyperparameters.items():
        default_type = type(current_value)
        new_value = form_data.get(key, current_value)
        try:
            if default_type == bool:
                app_state.hyperparameters[key] = str(new_value).lower() in ("true", "1", "on", "yes")
            else:
                app_state.hyperparameters[key] = default_type(new_value)
        except ValueError:
            pass # Ignore if the user sent an empty or unparseable string
    print(app_state.hyperparameters)
        # THE MAGIC HAPPENS HERE: 
    # Send a header back that tells HTMX to fire a custom JS event called "settingsSaved"
    response.headers["HX-Trigger"] = "settingsSaved"
    
    # Return a 204 No Content (empty body, success)
    response.status_code = 204
    return response


@app.post('/submit_race', response_class=HTMLResponse)
async def submit_race(request: Request):
    """Records race results, updates points, and advances to the next race."""
    form_data = await request.form()
    
    # Save state before mutating for the "Undo" feature
    save_history()
    global app_state
    current_race = app_state.races[app_state.current_race_idx]
    
    race_Length = len(current_race['players'])

    # Update points based on form submission
    for player_id_str, position_str in form_data.items():
        if player_id_str.startswith("pos_"):
            p_id = int(player_id_str.split("_")[1])
            pos = int(position_str)
            points_earned = POINTS_SYSTEM[race_Length].get(pos, 0)
            
            # Add points to player
            app_state.players[p_id]["points"] += points_earned
            # Store result in the race dict (optional, good for records)
            current_race.setdefault("results", {})[p_id] = pos
    
    # Advance race (but allow it to go ONE past the final index so we can see victory screen)
    if app_state.current_race_idx < len(app_state.races):
        app_state.current_race_idx += 1

    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"state": app_state, "leaderboard": get_sorted_leaderboard()}
    )


@app.post('/undo', response_class=HTMLResponse)
async def undo_last_action(request: Request):
    """Restores the previous state from the history stack."""
    global app_state
    if app_state.history:
        previous_state_dict = app_state.history.pop()
        # Restore state, maintaining the history stack
        history_backup = app_state.history
        app_state = TournamentState(**previous_state_dict)
        app_state.history = history_backup

    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"state": app_state, "leaderboard": get_sorted_leaderboard()}
    )


# --- ADDING / DROPPING PLAYER ENDPOINTS ---

@app.get('/components/new-add-player-row', response_class=HTMLResponse)
async def new_add_player_row(request: Request):
    """Returns a new row for the Add Player dialog."""
    return """
    <div class="add-player-row flex gap-3 items-center bg-gray-100 p-3 rounded-xl border-4 border-black shadow-[4px_4px_0_#000] relative group mb-4">
        <button type="button" onclick="this.closest('.add-player-row').remove(); checkGhostWarnings();" class="absolute -top-3 -right-3 bg-red-500 text-white w-8 h-8 rounded-full border-2 border-black font-bold shadow-[2px_2px_0_#000] opacity-0 group-hover:opacity-100 transition-opacity btn-tactile">
            <i class="fa-solid fa-xmark"></i>
        </button>
        <div class="flex-1">
            <label class="block text-xs font-black tracking-wider text-gray-600 mb-1">NAME</label>
            <input type="text" name="new_names" required class="w-full bg-white border-2 border-black rounded-lg p-2 font-bold focus:ring-4 focus:ring-yellow-400 outline-none">
        </div>
        <div class="w-20">
            <label class="block text-xs font-black tracking-wider text-gray-600 mb-1">MIN</label>
            <input type="number" name="min_races" min="0" required class="w-full bg-white border-2 border-black rounded-lg p-2 font-bold focus:ring-4 focus:ring-yellow-400 outline-none text-center">
        </div>
        <div class="w-20">
            <label class="block text-xs font-black tracking-wider text-gray-600 mb-1">MAX</label>
            <input type="number" name="max_races" min="0" required class="w-full bg-white border-2 border-black rounded-lg p-2 font-bold focus:ring-4 focus:ring-yellow-400 outline-none text-center">
        </div>
    </div>
    """


@app.get('/api/ghost-race-boundaries')
async def get_ghost_race_boundaries():
    A = backend_cache.incidence_matrix
    F = app_state.current_race_idx
    if A is None:
        return {"safe_min": 0, "safe_max": 99}
    safe_min, safe_max = dynamic_patching.get_ghost_safe_boundaries(A, F)
    # FastAPI automatically converts dictionaries to JSON responses!
    return {"safe_min": safe_min, "safe_max": safe_max}


@app.post('/api/add-players', response_class=HTMLResponse)
async def add_players(
    request: Request,
    new_names: list[str] = Form(...),
    min_races: list[int] = Form(...),
    max_races: list[int] = Form(...)
):
    global app_state
    global backend_cache
    save_history()

    boundaries = list(zip(min_races, max_races))
    F = app_state.current_race_idx
    
    # Run dynamic patching algorithm
    new_A = dynamic_patching.add_players(
        backend_cache.super_pools, 
        backend_cache.incidence_matrix, 
        F, 
        boundaries, 
        **app_state.hyperparameters
    )
    backend_cache.incidence_matrix = new_A
    print(new_A)
    print(new_A.shape)
    print(race_selection.get_co_occurrence_matrix(new_A))

    # Add new players to state
    start_id = max(app_state.players.keys()) + 1 if app_state.players else 0
    for i, name in enumerate(new_names):
        new_id = start_id + i
        app_state.players[new_id] = {"id": new_id, "name": name, "points": 0, "active": True}
    app_state.p_count += len(new_names)

    # Check to see if we need some more modifiers
    if new_A.shape[1] > len(backend_cache.shuffled_modifiers):
        # need to get some more modifiers
        num_more_needed = new_A.shape[1] - len(backend_cache.shuffled_modifiers)
        more_modifiers = modifier_shuffling.shuffle_out_modifiers(app_state.hyperparameters['seed'] + 1, num_more_needed, app_state.modifiers)
        backend_cache.shuffled_modifiers += more_modifiers

    # Delete races from F onwards
    app_state.races = app_state.races[:F]
    # Reconstruct the remaining races from the new incidence matrix starting at F
    ones_indices = race_ordering.get_indices_of_ones(new_A)
    # Add new races from F onwards
    for r in range(F, new_A.shape[1]):
        racer_ids = [int(p) for (p, col_r) in ones_indices if int(col_r) == r]
        app_state.races.append({'id': r, 'players': racer_ids, 'modifiers': backend_cache.shuffled_modifiers[r], 'results': {}})

    # Shuffle new modifiers for any newly added races
    # unmodified_races = [r for r in app_state.races if not r.get('modifiers')]
    # if unmodified_races:
    #     new_modifiers = modifier_shuffling.shuffle_out_modifiers(app_state.hyperparameters['seed'], len(unmodified_races), app_state.modifiers)
    #     for i, race in enumerate(unmodified_races):
    #         race['modifiers'] = new_modifiers[i]
        
    print(app_state.players)

    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"state": app_state, "leaderboard": get_sorted_leaderboard()}
    )


@app.post('/api/drop-players', response_class=HTMLResponse)
async def drop_players(request: Request, drop_ids: list[int] = Form(default=[])):
    global app_state
    global backend_cache
    if not drop_ids:
        # User submitted without selecting anyone
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"state": app_state, "leaderboard": get_sorted_leaderboard()}
        )
        
    save_history()
    F = app_state.current_race_idx

    new_A = dynamic_patching.drop_players(
        backend_cache.super_pools, 
        backend_cache.incidence_matrix, 
        F, 
        drop_ids, # Passing the raw matrix rows to the math function
        **app_state.hyperparameters
    )
    backend_cache.incidence_matrix = new_A
    print(new_A)
    print(new_A.shape)
    print(race_selection.get_co_occurrence_matrix(new_A))

    # Update active status and mapping
    for pid in drop_ids:
        if pid in app_state.players:
            app_state.players[pid]["active"] = False

    # Check to see if we need some more modifiers
    if new_A.shape[1] > len(backend_cache.shuffled_modifiers):
        # need to get some more modifiers
        num_more_needed = new_A.shape[1] - len(backend_cache.shuffled_modifiers)
        more_modifiers = modifier_shuffling.shuffle_out_modifiers(app_state.hyperparameters['seed'] + 1, num_more_needed, app_state.modifiers)
        backend_cache.shuffled_modifiers += more_modifiers

    # Delete the races that come during and after F
    app_state.races = app_state.races[:F]
    # Add any races from F onwards
    ones_indices = race_ordering.get_indices_of_ones(new_A)
    for r in range(F, new_A.shape[1]):
        racer_ids = [int(p) for (p, col_r) in ones_indices if int(col_r) == r]
        app_state.races.append({'id': r, 'players': racer_ids, 'modifiers': backend_cache.shuffled_modifiers[r], 'results': {}})

    # Shuffle new modifiers for any newly added races
    # unmodified_races = [r for r in app_state.races if not r.get('modifiers')]
    # if unmodified_races:
    #     new_modifiers = modifier_shuffling.shuffle_out_modifiers(
    #         app_state.hyperparameters['seed'], 
    #         len(unmodified_races), 
    #         app_state.modifiers
    #     )
    #     for i, race in enumerate(unmodified_races):
    #         race['modifiers'] = new_modifiers[i]

    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"state": app_state, "leaderboard": get_sorted_leaderboard()}
    )


# --- IMPORT / EXPORT ENDPOINTS ---
@app.post('/api/export-state')
async def export_state(
    request: Request,
    p_count: int = Form(...),
    n_races: int = Form(...),
    player_names: list[str] = Form(...)
):
    """Bundles current form data and state into a downloadable JSON file."""
    config = {
        "p_count": p_count,
        "n_races": n_races,
        "player_names": player_names,
        "hyperparameters": app_state.hyperparameters,
        "modifiers": app_state.modifiers
    }
    return Response(
        content=json.dumps(config, indent=4),
        media_type="application/json"
    )


@app.post('/api/import-state', response_class=HTMLResponse)
async def import_state(request: Request, file: UploadFile = File(...)):
    """Receives a JSON upload, updates the state, and triggers an HTMX full page re-render."""
    global app_state
    content = await file.read()
    try:
        config = json.loads(content)
        app_state.p_count = config.get("p_count", 5)
        app_state.n_races = config.get("n_races", 4)
        new_hyperparameters = config.get("hyperparameters", HYPERPARAMETER_DEFAULTS.copy())
        app_state.modifiers = config.get("modifiers", [])
        # Temporarily store player names so the setup screen can pre-fill the inputs
        player_names = config.get("player_names", [])
        app_state.players = {i: {"id": i, "name": name, "points": 0} for i, name in enumerate(player_names)}
    except Exception as e:
        print("Error loading config:", e)
        # Could return an error toast here in the future
    # add this for backwards compatibility
    for hyperparameter, default in HYPERPARAMETER_DEFAULTS.items():
        app_state.hyperparameters[hyperparameter] = new_hyperparameters.get(hyperparameter, default)
        
    # Re-render the full index template with the newly imported state
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"state": app_state, "leaderboard": []}
    )


# --- MODIFIER ENDPOINTS ---

@app.get('/components/new-modifier-row', response_class=HTMLResponse)
async def new_modifier_row(request: Request):
    return """
    <div class="modifier-row bg-purple-100 border-4 border-black rounded-2xl p-4 shadow-[4px_4px_0_#000] relative group">
        <!-- Optional: A tiny vanilla JS delete button for convenience before saving -->
        <button type="button" onclick="this.closest('.modifier-row').remove()" class="absolute -top-3 -right-3 bg-red-500 text-white w-8 h-8 rounded-full border-2 border-black font-bold shadow-[2px_2px_0_#000] opacity-0 group-hover:opacity-100 transition-opacity btn-tactile">
            <i class="fa-solid fa-xmark"></i>
        </button>
        <div class="flex flex-col gap-3">
            <!-- Top Row: Channel & Short Text -->
            <div class="flex gap-4 items-end">
                <div class="w-24">
                    <label class="block text-xs font-black tracking-wider text-gray-600 mb-1 uppercase">Channel</label>
                    <input type="number" name="channel" min="0" placeholder="0" class="w-full bg-white border-2 border-black rounded-xl p-2 font-bold focus:ring-4 focus:ring-yellow-400 outline-none shadow-[2px_2px_0_#000]">
                </div>
                <div class="flex-1">
                    <label class="block text-xs font-black tracking-wider text-gray-600 mb-1 uppercase">Short Name</label>
                    <input type="text" name="short_text" placeholder="e.g. Eat Grass" class="w-full bg-white border-2 border-black rounded-xl p-2 font-kart text-lg text-blue-600 focus:ring-4 focus:ring-yellow-400 outline-none shadow-[2px_2px_0_#000]">
                </div>
            </div>
            <!-- Bottom Row: Long Description -->
            <div>
                <label class="block text-xs font-black tracking-wider text-gray-600 mb-1 uppercase">Full Rule Description</label>
                <textarea name="description" rows="2" placeholder="Explain the exact rules of this modifier..." class="w-full bg-white border-2 border-black rounded-xl p-2 text-sm font-bold text-gray-800 focus:ring-4 focus:ring-yellow-400 outline-none shadow-[2px_2px_0_#000] resize-y"></textarea>
            </div>
        </div>
    </div>
    """

@app.post('/api/save-modifiers')
async def save_modifiers(
    request: Request,
    response: Response,
    # Use list[str] with defaults so it doesn't crash if the form is completely empty
    channel: list[str] = Form(default=[]),
    short_text: list[str] = Form(default=[]),
    description: list[str] = Form(default=[])):
    """Saves the dynamic list of modifiers."""
    modifiers_to_save = []
    # Iterate through the submitted arrays
    for i in range(len(channel)):
        # Skip if they left the short text or description completely blank
        if not short_text[i] and not description[i]:
            continue
        try:
            c_val = int(channel[i]) if channel[i] else 0
        except ValueError:
            c_val = 0  
        modifiers_to_save.append({
            "channel": c_val,
            "short_text": short_text[i],
            "description": description[i]
        })
    app_state.modifiers = modifiers_to_save
    # Tell HTMX to fire the "modifiersSaved" event on the frontend
    response.headers["HX-Trigger"] = "modifiersSaved"
    response.status_code = 204
    return response