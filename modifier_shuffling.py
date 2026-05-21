import jax
import jax.numpy as jnp
import numpy as np
from typing import Union


def collapse_modifiers_to_lists(modifiers_list:list[dict[str,Union[int,str]]]) -> dict[int,list[dict[str,str]]]:
    ret:dict[int,list[dict[str,str]]] = {}
    for modifier_dict in modifiers_list:
        ret.setdefault(modifier_dict['channel'], []).append({
            "short_text": modifier_dict["short_text"], "description": modifier_dict["description"]
        })
    return ret


def shuffle_out_modifiers(seed:int, number_of_races:int, modifiers_list:list[dict[str,Union[int,str]]]) -> list[list[dict[str,str]]]:
    """Returns a list (per race) of a list (per channel) of modifiers.  
    The returned list will be AT LEAST the size of modifiers_list. Not necessarily exact.
    You may trim it if you wish.
    """
    modifiers = collapse_modifiers_to_lists(modifiers_list)
    number_of_modifiers_per_channel = {channel:len(modifiers[channel]) for channel in modifiers.keys()}
    lcm = np.lcm.reduce(list(number_of_modifiers_per_channel.values()))
    # print(f"{lcm=}")
    modifier_list_length = int(jnp.ceil(number_of_races / lcm) * lcm)
    rng = jax.random.key(seed)
    shuffled_modifiers:dict[int,list[dict[str,str]]] = {channel:[] for channel in modifiers.keys()}
    # Shuffle each channel individually
    for channel in modifiers:
        num_modifiers = number_of_modifiers_per_channel[channel] # number of modifiers in the channel
        number_of_repeats_needed = int(modifier_list_length / num_modifiers) # how many times we gotta multiply the list
        # shuffle the channels as many times as we need repeats
        for repeat in range(number_of_repeats_needed):
            rng, shuffle_key = jax.random.split(rng)
            indices = jax.random.permutation(shuffle_key, num_modifiers)
            for idx in indices:
                shuffled_modifiers[channel].append(modifiers[channel][int(idx)])
    # transpose it such that we're grouping by race_idx rather than channel
    modifiers_per_race = [
        [
            shuffled_modifiers[channel][race_idx]
            for channel in modifiers
        ]
        for race_idx in range(modifier_list_length)
    ]
    return modifiers_per_race


if __name__ == '__main__':
    test_modifiers = [
        {'channel': 1, 'short_text': 'Eat Grass', 'description': 'test grass.'},
        {'channel': 1, 'short_text': 'Math Workout', 'description': 'test math.'},
        {'channel': 1, 'short_text': 'Condiment Shot', 'description': 'test condiment.'},
        {'channel': 1, 'short_text': 'Strong Man', 'description': 'test strong.'},
        {'channel': 2, 'short_text': 'Upside Down', 'description': 'test upside down.'},
        {'channel': 2, 'short_text': 'Double Drinks', 'description': 'test double.'},
        {'channel': 2, 'short_text': 'Partner Up', 'description': "test partner."},
        {'channel': 3, 'short_text': 'Test channel 3', 'description': 'test channel 3.'}
    ]
    number_of_races = 13
    seed = 42
    modifiers_per_race = shuffle_out_modifiers(seed, number_of_races, test_modifiers)
    print(f"{len(modifiers_per_race)=}")
    print('\n'.join(', '.join(d['short_text'] for d in race) for race in modifiers_per_race))
