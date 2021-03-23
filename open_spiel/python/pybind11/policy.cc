// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/python/pybind11/policy.h"

// Python bindings for policies and algorithms handling them.

#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/cfr_br.h"
#include "open_spiel/algorithms/deterministic_policy.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/explorative_cfr.h"
#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/algorithms/is_mcts.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/algorithms/outcome_sampling_mccfr.h"
#include "open_spiel/algorithms/tabular_best_response_mdp.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/policy.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/detail/common.h"

namespace open_spiel {
namespace {

using ::open_spiel::ActionsAndProbs;
using ::open_spiel::algorithms::Exploitability;
using ::open_spiel::algorithms::NashConv;
using ::open_spiel::algorithms::PlayerRegrets;
using ::open_spiel::algorithms::TabularBestResponse;
using ::open_spiel::algorithms::TabularBestResponseMDP;
using ::open_spiel::algorithms::TabularBestResponseMDPInfo;
using ::open_spiel::algorithms::ValuesMapT;
using ::open_spiel::algorithms::EpsilonCFRSolver;
using ::open_spiel::algorithms::ConditionalValuesEntry;
using ::open_spiel::algorithms::ConditionalValuesTable;
using ::open_spiel::algorithms::BRInfo;

namespace py = ::pybind11;
}  // namespace

void init_pyspiel_policy(py::module& m) {
  py::class_<TabularBestResponse>(m, "TabularBestResponse")
      .def(py::init<const open_spiel::Game&, int,
                    const std::unordered_map<std::string,
                                             open_spiel::ActionsAndProbs>&>())
      .def(py::init<const open_spiel::Game&, int, const open_spiel::Policy*>())
      .def("value",
  py::class_<TabularBestResponseMDPInfo>(m, "TabularBestResponseMDPInfo")
      .def_readonly("br_values", &TabularBestResponseMDPInfo::br_values)
      .def_readonly("br_policies", &TabularBestResponseMDPInfo::br_policies)
      .def_readonly("on_policy_values",
                    &TabularBestResponseMDPInfo::on_policy_values)
      .def_readonly("deviation_incentives",
                    &TabularBestResponseMDPInfo::deviation_incentives)
      .def_readonly("nash_conv", &TabularBestResponseMDPInfo::nash_conv)
      .def_readonly("exploitability",
                    &TabularBestResponseMDPInfo::exploitability);

  py::class_<TabularBestResponseMDP>(m, "TabularBestResponseMDP")
      .def(py::init<const open_spiel::Game&, const open_spiel::Policy&>())
      .def("compute_best_responses",  // Takes no arguments.
           &TabularBestResponseMDP::ComputeBestResponses)
      .def("compute_best_response",   // Takes one argument: Player max_player.
           &TabularBestResponseMDP::ComputeBestResponse, py::arg("max_player"))
      .def("nash_conv", &TabularBestResponseMDP::NashConv)
      .def("exploitability", &TabularBestResponseMDP::Exploitability);
  // Start Explorative CFR stuff
  py::class_<ConditionalValuesEntry> cventry(m, "ConditionalValuesEntry");
  cventry.def_readonly("player", &ConditionalValuesEntry::player)
      .def_readonly("info_state_key", &ConditionalValuesEntry::info_state_key)
      .def_readonly("value", &ConditionalValuesEntry::value)
      .def_readonly("max_qv_diff", &ConditionalValuesEntry::max_qv_diff)
      .def_readonly("legal_actions", &ConditionalValuesEntry::legal_actions)
      .def_readonly("action_values", &ConditionalValuesEntry::action_values);

  py::class_<BRInfo> br_info(m, "BRInfo");
  br_info.def_readonly("nash_conv", &BRInfo::nash_conv)
      .def_readonly("on_policy_values", &BRInfo::on_policy_values)
      .def_readonly("deviation_incentives", &BRInfo::deviation_incentives)
      .def_readonly("cvtables", &BRInfo::cvtables);

  py::class_<ConditionalValuesTable>(m, "ConditionalValuesTable")
      .def(py::init<int>(), py::arg("num_players"))
      .def("add_entry", &ConditionalValuesTable::add_entry)
      .def("num_players", &ConditionalValuesTable::num_players)
      .def("max_qv_diff", &ConditionalValuesTable::max_qv_diff)
      .def("avg_qv_diff", &ConditionalValuesTable::avg_qv_diff)
      .def("import", &ConditionalValuesTable::Import)
      .def("table", &ConditionalValuesTable::table);

  py::class_<open_spiel::algorithms::EpsilonCFRSolver>(m, "EpsilonCFRSolver")
      .def(py::init<const Game&, double>(), py::arg("game"),
           py::arg("epsilon"))
      .def("epsilon", &EpsilonCFRSolver::epsilon)
      .def("set_epsilon", &EpsilonCFRSolver::SetEpsilon)
      .def("evaluate_and_update_policy",
           &EpsilonCFRSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &EpsilonCFRSolver::CurrentPolicy)
      .def("average_policy", &EpsilonCFRSolver::AveragePolicy)
      .def("tabular_average_policy", &EpsilonCFRSolver::TabularAveragePolicy);

  m.def("merge_tables", &open_spiel::algorithms::MergeTables);

  m.def("nash_conv_with_eps", &open_spiel::algorithms::NashConvWithEps);

  m.def("expected_returns",
        py::overload_cast<const State&, const std::vector<const Policy*>&, int,
                          bool, ValuesMapT*>(
                              &open_spiel::algorithms::ExpectedReturns),
        "Computes the undiscounted expected returns from a depth-limited "
        "search.",
        py::arg("state"),
        py::arg("policies"),
        py::arg("depth_limit"),
        py::arg("use_infostate_get_policy"),
        py::arg("prob_cut_threshold") = 0.0);

  m.def("expected_returns",
        py::overload_cast<const State&, const Policy&, int,
                          bool, float>(
                              &open_spiel::algorithms::ExpectedReturns),
        "Computes the undiscounted expected returns from a depth-limited "
        "search.",
        py::arg("state"),
        py::arg("joint_policy"),
        py::arg("depth_limit"),
        py::arg("use_infostate_get_policy"),
        py::arg("prob_cut_threshold") = 0.0);

  m.def("exploitability",
        py::overload_cast<const Game&, const Policy&>(&Exploitability),
        "Returns the sum of the utility that a best responder wins when when "
        "playing against 1) the player 0 policy contained in `policy` and 2) "
        "the player 1 policy contained in `policy`."
        "This only works for two player, zero- or constant-sum sequential "
        "games, and raises a SpielFatalError if an incompatible game is passed "
        "to it.");

  m.def(
      "exploitability",
      py::overload_cast<
          const Game&, const std::unordered_map<std::string, ActionsAndProbs>&>(
          &Exploitability),
      "Returns the sum of the utility that a best responder wins when when "
      "playing against 1) the player 0 policy contained in `policy` and 2) "
      "the player 1 policy contained in `policy`."
      "This only works for two player, zero- or constant-sum sequential "
      "games, and raises a SpielFatalError if an incompatible game is passed "
      "to it.");

  m.def("nash_conv",
        py::overload_cast<const Game&, const Policy&, bool>(&NashConv),
        "Calculates a measure of how far the given policy is from a Nash "
        "equilibrium by returning the sum of the improvements in the value "
        "that each player could obtain by unilaterally changing their strategy "
        "while the opposing player maintains their current strategy (which "
        "for a Nash equilibrium, this value is 0). The third parameter is to "
        "indicate whether to use the Policy::GetStatePolicy(const State&) "
        "instead of Policy::GetStatePolicy(const std::string& info_state) for "
        "computation of the on-policy expected values.",
        py::arg("game"), py::arg("policy"),
        py::arg("use_state_get_policy") = false);

  m.def(
      "nash_conv",
      py::overload_cast<
          const Game&, const std::unordered_map<std::string, ActionsAndProbs>&>(
          &NashConv),
      "Calculates a measure of how far the given policy is from a Nash "
      "equilibrium by returning the sum of the improvements in the value "
      "that each player could obtain by unilaterally changing their strategy "
      "while the opposing player maintains their current strategy (which "
      "for a Nash equilibrium, this value is 0).");

  m.def(
      "player_regrets",
      py::overload_cast<
          const Game&, const Policy&, bool>(
          &PlayerRegrets),
      "Return regret vector");

  m.def("num_deterministic_policies",
        &open_spiel::algorithms::NumDeterministicPolicies,
        "Returns number of determinstic policies in this game for a player, "
        "or -1 if there are more than 2^64 - 1 policies.");

  m.def("to_joint_tabular_policy", &open_spiel::ToJointTabularPolicy,
        "Returns a merged tabular policy from a list of TabularPolicy. The "
        "second argument is a bool which, if true, checks that there is no "
        "overlap among all the policies.");
}
}  // namespace open_spiel
