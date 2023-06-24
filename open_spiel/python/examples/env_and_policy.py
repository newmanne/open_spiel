from dataclasses import dataclass
from open_spiel.python.rl_environment import Environment
from open_spiel.python.rl_agent_policy import JointRLAgentPolicy
from typing import List
import pyspiel

@dataclass
class EnvAndPolicy:
    env: Environment
    agents: List
    game: pyspiel.Game

    def make_policy(self, agents=None, string_only=False) -> JointRLAgentPolicy:
        if agents is None:
            agents = self.agents
        agent_dict = {agent.player_id: agent for agent in agents}
        return JointRLAgentPolicy(self.game, agent_dict, False, string_only=string_only)
