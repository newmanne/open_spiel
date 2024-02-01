# Functions for comparing on-path behaviour

from open_spiel.python.algorithms.exploitability import _state_values

def traverse(state, policy, cfr_policy, br_policies, br_player, follow_br=False, depth=0):
    """
    TODO: document

    see 2023-11-02-MixingExample.ipynb for example usage
    """
    def print_depth(s):
        if depth == 0:
            prefix = ''
        else:
            prefix = '--' * (depth-1) + '- '
        print(f'{prefix}{s}')
        
    if state.is_terminal():
        player_return = state.returns()[br_player]
        print_depth(f'Terminal state: utility = {player_return:.2f}')
        return
        
    # compute values
    player_value = _state_values(state, 2, policy)[br_player]
    br_value = br_policies[br_player].value(state)
    print_depth(f'Values: mode = {player_value:.2f}; BR = {br_value:.2f}')

    if state.is_chance_node():
        chance_outcomes = state.chance_outcomes()
        # print_depth(f'Chance node')
        for (action, action_prob) in chance_outcomes:
            print_depth(f'Chance action {action} (p={action_prob:.2f}):')
            traverse(state.child(action), policy, cfr_policy, br_policies, br_player, follow_br, depth+1)
            
    else:
        player = state.current_player()
        action_probs = policy._agents[player].policy.action_probabilities(state)
        mode_action = max(action_probs, key=action_probs.get)
        infostate_string = state.information_state_string()
        regrets, avg_policy, visits, _ = cfr_policy._infostates[infostate_string]
        avg_policy_visits = avg_policy.sum().round()
        regret_visits = visits - avg_policy_visits
        
        
        if player == br_player:
            br_action_probs = br_policies[player].action_probabilities(state)
            br_action = max(br_action_probs, key=br_action_probs.get)
        
            if br_action != mode_action:
                print_depth(f'[ BR != mode ({br_action} != {mode_action})')
                print_depth(f'[ modal value: {_state_values(state.child(br_action), 2, policy)[br_player]:5.2f}')
                print_depth(f'[ CFR stats:')
                print_depth(f'[ * regrets: {regrets.round(2)}')
                print_depth(f'[ * avg_policy: {avg_policy.round(2)}')
                print_depth(f'[ * regret visits: {regret_visits}')
                print_depth(f'[ * avg policy visits: {avg_policy_visits}')
        
        if player == br_player and follow_br:
            print_depth(f'P{player} BR action = {br_action} (policy p = {action_probs[br_action]:.2f}; {regret_visits} regret updates; regrets = {regrets.round(2)}):')       
            traverse(state.child(br_action), policy, cfr_policy, br_policies, br_player, follow_br, depth+1)
            
        else:
            print_depth(f'P{player} modal action = {mode_action} (p = {action_probs[mode_action]:.2f}; {regret_visits} regret updates; regrets = {regrets.round(2)}):')
            traverse(state.child(mode_action), policy, cfr_policy, br_policies, br_player, follow_br, depth+1)