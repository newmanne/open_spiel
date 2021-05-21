# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example algorithm to get all states from a game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO: Use logs to avoid so much small number multiplication?

def _get_subgames_states(state, all_states, depth_limit, depth,
                         include_terminals, to_string,
                         policy, curr_prob=1.0, max_depth=0):
  """Extract non-chance states for a subgame into the all_states dict."""
  if state.is_terminal():
    if include_terminals:
      # Include if not already present and then terminate recursion.
      state_str = to_string(state)
      if state_str not in all_states:
        all_states[state_str] = dict(state=state.clone(), prob=curr_prob)
      else:
        all_states[state_str]['prob'] += curr_prob
    return max_depth

  if depth > depth_limit >= 0:
    return max_depth

  if not state.is_chance_node():
    # Add only if not already present
    info_state_string = state.information_state_string()

    if info_state_string not in all_states:
      all_states[info_state_string] = dict(state=state.clone(), prob=curr_prob)
    else:
      all_states[info_state_string]['prob'] += curr_prob
      
  if policy is not None:
    action_probabilities = policy.action_probabilities(state) if not state.is_chance_node() else dict(state.chance_outcomes())
  else:
    action_probabilities = dict()
  for action in state.legal_actions():
    prob = curr_prob * action_probabilities.get(action, 1)
    state_for_search = state.child(action)
    md = _get_subgames_states(state_for_search, all_states, depth_limit, depth + 1,
                         include_terminals, to_string,
                         policy, prob, max(depth + 1, max_depth))
    md = max(md, max_depth)
  return md
 

def get_all_info_states_with_policy(game,
                   depth_limit=-1,
                   include_terminals=True,
                   to_string=lambda s: s.history_str(),
                   policy=None):
  '''to_string: collapse terminals that you want treated the same'''
                  
  # Get the root state.
  state = game.new_initial_state()
  all_states = dict()

  # Then, do a recursive tree walk to fill up the map.
  max_depth = _get_subgames_states(
      state=state,
      all_states=all_states,
      depth_limit=depth_limit,
      depth=0,
      include_terminals=include_terminals,
      to_string=to_string,
      policy=policy,
      curr_prob=1.0,
      max_depth=1)

  print(f'Max Depth: {max_depth}')

  if not all_states:
    raise ValueError("GetSubgameStates returned 0 states!")

  return all_states
