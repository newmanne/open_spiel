"""
More efficient best response calcuations.
"""

NEGATIVE_INF = float("-inf")
NO_DEVIATION_LIMIT = -1

# TODO: cache subgame values?
# Our auctions have many states that are equivalent except for their history;
# if the opponent's strategy doesn't depend on the history (e.g., neither were seen during training), 
# we only need to compute the value once for these equivalent subgames.
# 
# something along these lines...
# from cachetools import LRUCache, cached
# def subgame_cache_key(state, *args, **kwargs):
#     return state.key_without_history()
# @cached(cache=LRUCache(maxsize=10000), key=subgame_cache_key)
# def _compute_br_value(state, ...):
# 
# Greg: I haven't done this because I'm worried about value_lower_bound and deviations_allowed interacting poorly with the cache.

def compute_br_value(
    state, 
    player_id, 
    policy, 
):
    """Computes the best response value for a given player.
    
    Args:
        state: The current state of the game.
        player_id: The player id of the best-responder.
        policy: A `policy.Policy` object.
        prune_subgames: Whether to prune subtrees where we can't beat the best known value (default: True).
        value_lower_bound: A lower bound on the best response value (default: -inf).
        deviations_allowed: The number of deviations allowed from the policy's modal action (default: -1, no limit).
        accuracy_threshold: Improvement required over the lower bound to consider an action (default: 0, exact computation).

    Returns:
        The value of the best response.
    """

    if state.is_terminal():
        return state.player_return(player_id)
    
    elif state.is_chance_node():
        return sum(prob * compute_br_value(state.child(action), player_id, policy) for action, prob in state.chance_outcomes())

        value = 0
        chance_outcomes = state.chance_outcomes()
        for i in range(len(chance_outcomes)):
            action, prob = chance_outcomes[i]
            # can only prune in the last chance node
            if i == len(chance_outcomes) - 1:
                # need prob * child value + current value > value_lower_bound
                # or: child value > (value_lower_bound - current value) / prob 
                remaining_value_needed = (value_lower_bound - value) / prob
                value += prob * compute_br_value(state.child(action), player_id, policy, remaining_value_needed, accuracy_threshold)
            else:
                value += prob * compute_br_value(state.child(action), player_id, policy, NEGATIVE_INF, accuracy_threshold)
        return value 
    
    elif state.is_simultaneous_node():
        raise NotImplementedError("Simultaneous games not supported.")

    elif state.current_player() != player_id:
        # if the opponent's strategy is deterministic, we can pass the best known value down
        opponent_strategy = policy.action_probabilities(state)
        if max(opponent_strategy.values()) == 1.0: # TODO: check if close to 1 instead?
            # find the action 
            action = max(opponent_strategy, key=opponent_strategy.get)
            return compute_br_value(state.child(action), player_id, policy)

        # otherwise, we can't prune
        else:
            raise NotImplementedError("Opponent strategies must be deterministic.")
        
    else: # current player is the best responder
        legal_actions = state.legal_actions()
        best_value = NEGATIVE_INF
        best_action = None

        for action in legal_actions:
            value = compute_br_value(state.child(action), player_id, policy)
            if value > best_value:
                best_value = value
                best_action = action

        print(state.information_state_string(), best_action, best_value)

        return best_value
