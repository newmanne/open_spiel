from open_spiel.python.rl_agent import AbstractAgent, StepOutput
from cachetools import cached, LRUCache, TTLCache
from cachetools.keys import hashkey
from open_spiel.python.examples.ubc_utils import single_action_result, fast_choice
import numpy as np
from open_spiel.python import rl_agent
from absl import logging


# TODO: Is all of this meta magic slowing you down? Or is it not worth bothering
BLANK_OUTPUT = rl_agent.StepOutput(action=None, probs=[])

class AgentDecorator(AbstractAgent):

    _agent: AbstractAgent = None

    def __init__(self, agent: AbstractAgent) -> None:
        self._agent = agent
        self.agent_attributes = [attribute for attribute in self._agent.__dict__.keys()]
        self.agent_methods = [m for m in dir(self._agent) if not m.startswith('_') and m not in self.agent_attributes]

    def __getattr__(self, func):
        if func in self.agent_methods:
            def method(*args):
                return getattr(self._agent, func)(*args)
            return method
        elif func in self.agent_attributes:
            return getattr(self._agent, func)
        else:
            # For nesting decorators
            if isinstance(self._agent, AgentDecorator):
                return self._agent.__getattr__(func)
            raise AttributeError(func)

    @property
    def agent(self) -> AbstractAgent:
        return self._agent

    def step(self, time_step, is_evaluation=False):
        return self._agent.step(time_step, is_evaluation=is_evaluation)


class CachingAgentDecorator(AgentDecorator):

    def __init__(self, agent):
        super().__init__(agent)
        self.cache = LRUCache(maxsize=50_000)

    def step(self, time_step, is_evaluation=False):
        key = hashkey(tuple(time_step.observations["info_state"][self.player_id]))
        val = self.cache.get(key)
        if val is not None:
            # Reselect action randomly, but use cached probs
            if val == BLANK_OUTPUT:
                return val # Terminal node
            else:
                action = fast_choice(range(len(val.probs)), val.probs)
                return rl_agent.StepOutput(action=action, probs=val.probs)
        else:
            output = self.agent.step(time_step, is_evaluation=is_evaluation)
            self.cache[key] = output
            return output


class TakeSingleActionDecorator(AgentDecorator):
    '''Sometimes you have to write lots of code to avoid telling an agent to take literally the only action it can. This avoids that'''

    def __init__(self, agent, num_actions):
        super().__init__(agent)
        self.num_actions = num_actions

    def step(self, time_step, is_evaluation=False):
        if isinstance(time_step, list):
            return [self.step(time_step[i], is_evaluation=is_evaluation) for i in range(len(time_step))]

        legal_actions = time_step.observations["legal_actions"][self.player_id]
        if len(legal_actions) == 1:
            return single_action_result(legal_actions, self.num_actions, as_output=True)
        return self._agent.step(time_step, is_evaluation=is_evaluation)



class TremblingAgentDecorator(AgentDecorator):

    def __init__(self, agent, tremble_prob=0.01, tremble_model=None):
        super().__init__(agent)
        self.tremble_model = tremble_model
        self.tremble_prob = tremble_prob
        self.curr_tremble_prob = self.tremble_prob

    def _step(self, regular_output, time_step):
        regular_probs = np.asarray(regular_output.probs) # Might be torch
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        num_legal_actions = len(legal_actions)

        if len(regular_probs) == num_legal_actions:
            # We only see probs for legal actions
            uniform_policy = np.ones(len(regular_probs), dtype=np.float64) / num_legal_actions
            sample_policy = self.curr_tremble_prob * uniform_policy + (1.0 - self.curr_tremble_prob) * regular_probs
            return StepOutput(action=fast_choice(legal_actions, sample_policy), probs=sample_policy)
        else:
            # Illegal actions are NOT BEING MASKED OUT and may have positive probabilities
            uniform_policy = np.zeros(len(regular_probs), dtype=np.float64)
            uniform_policy[legal_actions] = 1. / num_legal_actions
            sample_policy = self.curr_tremble_prob * uniform_policy + (1.0 - self.curr_tremble_prob) * regular_probs
            s = sample_policy[legal_actions] / sample_policy[legal_actions].sum()
            return StepOutput(action=fast_choice(legal_actions, s), probs=s)

    def step(self, time_step, is_evaluation=False):
        regular_output = self._agent.step(time_step, is_evaluation=is_evaluation)
        if isinstance(regular_output, list):
            o = []
            for ro, ts in zip(regular_output, time_step):
                o.append(self._step(ro, ts))
            return o
        else:
            return self._step(regular_output, time_step)


    def anneal_learning_rate(self, update, num_total_updates):
        # Super hack
        self._agent.anneal_learning_rate(update, num_total_updates)
        frac = 1.0 - (update / num_total_updates)
        self.curr_tremble_prob = frac * self.tremble_prob
        print(f"Annealed to {self.curr_tremble_prob}")


class ModalAgentDecorator(AgentDecorator):

    def __init__(self, agent):
        super().__init__(agent)

    def step(self, time_step, is_evaluation=False):
        if isinstance(time_step, list):
            return [self.step(time_step[i], is_evaluation=is_evaluation) for i in range(len(time_step))]
        
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        num_legal_actions = len(legal_actions)
        regular_output = self._agent.step(time_step, is_evaluation=is_evaluation)
        regular_probs = regular_output.probs
        new_probs = np.zeros_like(regular_probs, dtype=np.float64)

        if len(regular_probs) == num_legal_actions:
            # We only see probs for legal actions
            legal_action_idx = np.argmax(regular_probs)
            action = legal_actions[legal_action_idx]
            new_probs[legal_action_idx] = 1.
            return StepOutput(action=action, probs=new_probs)
        else:
            # We see probs for all actions; illegal actions are NOT BEING MASKED OUT and may have positive probabilities
            action = legal_actions[np.argmax(regular_probs[legal_actions])]
            new_probs[action] = 1.
            return StepOutput(action=action, probs=new_probs)
    

class UniformRestrictedNashResponseAgent(AgentDecorator):

    def __init__(self, exploit_agent, trained_agents, agent_selector):
        super().__init__(exploit_agent)
        self.agent_selector = agent_selector
        self.trained_agents = trained_agents

    def step(self, time_step, is_evaluation=False):
        agent_id = self.agent_selector.get_episode_agent()
        if agent_id == -1:
            return self._agent.step(time_step, is_evaluation=is_evaluation)
        else:
            return self.trained_agents[agent_id].step(time_step, is_evaluation=True)

class UniformRestrictedNashResponseAgentSelector:

    def __init__(self, n, num_players, exploit_prob=0.5, iterate_br=True, rnr_player_id =0):
        self.exploit_prob = exploit_prob
        self.n = n
        self._episode_agent = -1
        self.iterate_br = iterate_br
        self.rnr_player_id = rnr_player_id
        self.num_player = num_players

    def new_episode(self, ep):
        if self.iterate_br and ep % self.num_player != self.rnr_player_id:
            self._episode_agent = -1 # If the learning player is not the RNR agent, it should actually learn (along with all the other agents?)
        else:
            options = [-1] + list(range(self.n))
            x = (1 - self.exploit_prob) / self.n
            probs = [self.exploit_prob] + [x] * self.n
            self._episode_agent = fast_choice(options, probs)

    def get_episode_agent(self):
        return self._episode_agent