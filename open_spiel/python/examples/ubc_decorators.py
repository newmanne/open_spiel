from open_spiel.python.rl_agent import AbstractAgent
from cachetools import cached, LRUCache, TTLCache
from cachetools.keys import hashkey

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
            raise AttributeError

    @property
    def agent(self) -> AbstractAgent:
        return self._agent

    def step(self, time_step, is_evaluation=False):
        return self._agent.step(time_step, is_evaluation=is_evaluation)


class CachingAgentDecorator(AgentDecorator):

    def __init__(self, agent):
        super().__init__(agent)
        self.cache = LRUCache(maxsize=5000)

    def step(self, time_step, is_evaluation=False):
        key = hashkey(tuple(time_step.observations["info_state"][self.player_id]))
        if key in self.cache:
            return self.cache[key]
        else:
            output = self.agent.step(time_step, is_evaluation=is_evaluation)
            self.cache[key] = output
            return output
