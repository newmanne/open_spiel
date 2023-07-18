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

"""Visualizing game trees with graphviz.

GameTree builds a `pygraphviz.AGraph` reprensentation of the game tree. The
resulting tree can be directly visualized in Jupyter notebooks or Google Colab
via SVG plotting - or written to a file by calling `draw(filename, prog="dot")`.

See `examples/treeviz_example.py` for a more detailed example.

This module relies on external dependencies, which need to be installed before
use. On a debian system follow these steps:
```
sudo apt-get install graphviz libgraphviz-dev
pip install pygraphviz
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pyspiel
from absl import logging

# pylint: disable=g-import-not-at-top
try:
  import pygraphviz
except (ImportError, Exception) as e:
  raise ImportError(
      str(e) + "\nPlease make sure to install the following dependencies:\n"
      "sudo apt-get install graphviz libgraphviz-dev\n"
      "pip install pygraphviz")
# pylint: enable=g-import-not-at-top

from open_spiel.python.examples.ubc_utils import *

_PLAYER_SHAPES = {0: "ellipse", 1: "ellipse"}
_PLAYER_COLORS = {-1: "black", 0: "blue", 1: "red", 2: "orange", 3: "green"}
_FONTSIZE = 8
_WIDTH = _HEIGHT = 0.25
_ARROWSIZE = .5
_MARGIN = 0.01


def default_node_decorator(state, **kwargs):
  """Decorates a state-node of the game tree.

  This method can be called by a custom decorator to prepopulate the attributes
  dictionary. Then only relevant attributes need to be changed, or added.

  Args:
    state: The state.

  Returns:
    `dict` with graphviz node style attributes.
  """
  player = state.current_player()
  attrs = {
      "label": "",
      "fontsize": _FONTSIZE,
      "width": _WIDTH,
      "height": _HEIGHT,
      "margin": _MARGIN
  }
  if state.is_terminal():
    attrs["label"] = ", ".join(map(str, state.returns()))
    attrs["shape"] = "diamond"
  elif state.is_chance_node():
    attrs["shape"] = "point"
    attrs["width"] = _WIDTH / 2.
    attrs["height"] = _HEIGHT / 2.
  else:
    attrs["label"] = str(state.information_state_string())
    attrs["shape"] = _PLAYER_SHAPES.get(player, "ellipse")
    attrs["color"] = _PLAYER_COLORS.get(player, "black")
  return attrs


def default_edge_decorator(parent, unused_child, action, **kwargs):
  """Decorates a state-node of the game tree.

  This method can be called by a custom decorator to prepopulate the attributes
  dictionary. Then only relevant attributes need to be changed, or added.

  Args:
    parent: The parent state.
    unused_child: The child state, not used in the default decorator.
    action: `int` the selected action in the parent state.

  Returns:
    `dict` with graphviz node style attributes.
  """
  player = parent.current_player()
  attrs = {
      "label": " " + parent.action_to_string(player, action),
      "fontsize": _FONTSIZE,
      "arrowsize": _ARROWSIZE
  }
  attrs["color"] = _PLAYER_COLORS.get(player, "black")
  return attrs


def make_policy_decorators(policy):
    def edge_weight_by_policy_decorator(parent_state, unused_child, action, **kwargs):
        attrs = default_edge_decorator(parent_state, unused_child, action)  # get default attributes
        if int(parent_state.current_player()) < 0:
            return attrs
        
        if '@ $0' in attrs['label']:
            attrs['label'] = 'Drop Out'
        
        action_prob = policy.action_probabilities(parent_state)[action]
        attrs['arrowsize'] = action_prob * attrs['arrowsize']
        attrs['penwidth'] = action_prob
        attrs['label'] = f'[{action_prob:.2f}] {attrs["label"]}'
        
        # legal_actions = parent_state.legal_actions()
        # parent_tensor = parent_state.information_state_tensor()
        
        # Grab from encoding
        # TODO: These really don't need to be recomputed...
        # n_actions = parent_state.num_distinct_actions()
        # n_players = parent_state.num_players()
        # cpi = clock_profit_index(n_players, n_actions)
        # profits = parent_tensor[cpi:cpi + n_actions]
        # legal_profits = [profits[i] for i in legal_actions]
        # straightforward_action = legal_actions[np.argmax(legal_profits)]
        
        # if straightforward_action == action:
        #   attrs['label'] += ' (Straightforward)'

#         print(attrs, parent_state.current_player())
        return attrs

    def node_weight_by_policy_decorator(state, **kwargs):
        attrs = default_node_decorator(state)
        if state.is_terminal():
            return_list = eval(attrs['label'])
            attrs['label'] = ', '.join([f'{x:.2f}' for x in return_list])

            if 'state_prob' in kwargs:
              attrs['label'] += f'\n({(kwargs["state_prob"]*100):.3f}% reach)'
              scale_factor = kwargs['state_prob'] * 5000
              if scale_factor > 5:
                scale_factor = 5
              if scale_factor < 1:
                scale_factor = 1
              # attrs["width"] = _WIDTH * scale_factor
              attrs["height"] = _HEIGHT * scale_factor

        # TODO: If terminal, report allocation
        # {'label': 'Current player: 0\np0v125, 125b150\n', 'fontsize': 8, 'width': 0.25, 'height': 0.25, 'margin': 0.01, 'shape': 'square', 'color': 'blue'}
        #         print(attrs)
        return attrs

    return node_weight_by_policy_decorator, edge_weight_by_policy_decorator

class GameTree(pygraphviz.AGraph):
  """Builds `pygraphviz.AGraph` of the game tree.

  Attributes:
    game: A `pyspiel.Game` object.
    depth_limit: Maximum depth of the tree. Optional, default=-1 (no limit).
    node_decorator: Decorator function for nodes (states). Optional, default=
      `treeviz.default_node_decorator`.
    edge_decorator: Decorator function for edges (actions). Optional, default=
      `treeviz.default_edge_decorator`.
    group_terminal: Whether to display all terminal states at same level,
      default=False.
    group_infosets: Whether to group infosets together, default=False.
    group_pubsets: Whether to group public sets together, default=False.
    target_pubset: Whether to group all public sets "*" or a specific one.
    infoset_attrs: Attributes to style infoset grouping.
    pubset_attrs: Attributes to style public set grouping.
    kwargs: Keyword arguments passed on to `pygraphviz.AGraph.__init__`.
  """

  def __init__(self,
               game=None,
               depth_limit=-1,
               node_decorator=default_node_decorator,
               edge_decorator=default_edge_decorator,
               group_terminal=False,
               group_infosets=False,
               group_pubsets=False,
               target_pubset="*",
               infoset_attrs=None,
               pubset_attrs=None,
               policy=None,
               state_prob_limit=None,
               action_prob_limit=None,
               **kwargs):
    kwargs["directed"] = kwargs.get("directed", True)
    super(GameTree, self).__init__(**kwargs)

    # We use pygraphviz.AGraph.add_subgraph to cluster nodes, and it requires a
    # default constructor. Thus game needs to be optional.
    if game is None:
      return

    self.game = game
    self._node_decorator = node_decorator
    self._edge_decorator = edge_decorator
    self.policy = policy
    if self.policy is None and (state_prob_limit or action_prob_limit):
      raise ValueError("Must supply policy to use state/action prob limits!")

    self.state_prob_limit = state_prob_limit

    self._group_infosets = group_infosets
    self._group_pubsets = group_pubsets
    if self._group_infosets:
      if not self.game.get_type().provides_information_state_string:
        raise RuntimeError(
            "Grouping of infosets requested, but the game does not "
            "provide information state string.")
    if self._group_pubsets:
      if not self.game.get_type().provides_factored_observation_string:
        raise RuntimeError(
            "Grouping of public sets requested, but the game does not "
            "provide factored observations strings.")

    self._infosets = collections.defaultdict(lambda: [])
    self._pubsets = collections.defaultdict(lambda: [])
    self._terminal_nodes = []

    root = game.new_initial_state()
    self.add_node(self.state_to_str(root), **self._node_decorator(root))
    logging.info("Building tree...")
    self._build_tree(root, 0, depth_limit, state_prob_limit=state_prob_limit, state_prob=1., action_prob_limit=action_prob_limit)
    logging.info("Built tree!")

    for (player, info_state), sibblings in self._infosets.items():
      cluster_name = "cluster_{}_{}".format(player, info_state)
      self.add_subgraph(sibblings, cluster_name,
                        **(infoset_attrs or {
                            "style": "dashed"
                        }))

    for pubset, sibblings in self._pubsets.items():
      if target_pubset == "*" or target_pubset == pubset:
        cluster_name = "cluster_{}".format(pubset)
        self.add_subgraph(sibblings, cluster_name,
                          **(pubset_attrs or {
                              "style": "dashed"
                          }))

    if group_terminal:
      self.add_subgraph(self._terminal_nodes, rank="same")

  def state_to_str(self, state):
    """Unique string representation of a state.

    Args:
      state: The state.

    Returns:
      String representation of state.
    """
    assert not state.is_simultaneous_node()
    # AGraph nodes can't have empty string == None as a key, thus we prepend " "
    return " " + state.history_str()

  def _build_tree(self, state, depth, depth_limit, state_prob_limit=None, action_prob_limit=None, state_prob=1.0):
    """Recursively builds the game tree."""
    state_str = self.state_to_str(state)

    if state.is_terminal():
      self._terminal_nodes.append(state_str)
      return
    if depth > depth_limit >= 0:
      return

    if self.policy:
      action_probs = self.policy.action_probabilities(state) if not state.is_chance_node() else dict(state.chance_outcomes())
      # if not state.is_chance_node():
      #   print(state.information_state_string(), action_probs)

    for action in state.legal_actions():
      kwargs = dict(state_prob_limit=state_prob_limit, state_prob=state_prob, action_prob_limit=action_prob_limit)
      if self.policy:
        action_prob = action_probs[action]
        if action_prob_limit is not None and action_prob < action_prob_limit:
          # This action is never chosen, so we'll just not draw it or it's descendents
          continue
        kwargs['state_prob'] *= action_prob

        if state_prob_limit is not None and kwargs['state_prob'] < state_prob_limit:
          # The probability of reaching this state is so low, we quit
          continue

      child = state.child(action)
      child_str = self.state_to_str(child)
      self.add_node(child_str, **self._node_decorator(child, **kwargs))
      self.add_edge(state_str, child_str, **self._edge_decorator(state, child, action, **kwargs))

      if self._group_infosets and not child.is_chance_node() and not child.is_terminal():
        player = child.current_player()
        info_state = child.information_state_string()
        self._infosets[(player, info_state)].append(child_str)

      if self._group_pubsets:
        pub_obs_history = str(pyspiel.PublicObservationHistory(child))
        self._pubsets[pub_obs_history].append(child_str)

      self._build_tree(child, depth + 1, depth_limit, **kwargs)

  def _repr_svg_(self):
    """Allows to render directly in Jupyter notebooks and Google Colab."""
    if not self.has_layout:
      self.layout(prog="dot")
    return self.draw(format="svg").decode(self.encoding)
