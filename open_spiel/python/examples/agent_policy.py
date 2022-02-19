from open_spiel.python import rl_environment, policy

class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, best_response_mode):
        game = env.game
        player_ids = list(range(len(nfsp_policies)))
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._best_response_mode = best_response_mode
        self._obs = {"info_state": [None] * len(player_ids), "legal_actions": [None] * len(player_ids)}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(observations=self._obs, rewards=None, discounts=None, step_type=None)

        p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict

    def save(self):
        output = dict()
        for player, policy in enumerate(self._policies):
            output[player] = policy.save()
        return output

    def restore(self, restore_dict):
        for player, policy in enumerate(self._policies):
            policy.restore(restore_dict[player])