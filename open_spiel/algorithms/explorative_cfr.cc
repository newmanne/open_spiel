// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/explorative_cfr.h"

#include "open_spiel/algorithms/best_response.h"

namespace open_spiel {
namespace algorithms {

EpsilonCFRSolver::EpsilonCFRSolver(const Game& game, double initial_epsilon)
  : CFRSolverBase(game,
                  /*alternating_updates=*/true,
                  /*linear_averaging=*/true,
                  /*regret_matching_plus=*/true),
    epsilon_(initial_epsilon) {}

void EpsilonCFRSolver::EvaluateAndUpdatePolicy() {
  ++iteration_;

  for (int player = 0; player < game_->NumPlayers(); player++) {
    ComputeCounterFactualRegret(*root_state_, player, root_reach_probs_,
                                nullptr);
  }

  ApplyRegretMatchingPlusReset();
  ApplyEpsilonRegretMatching();

  // epsilon_ -=
}

void EpsilonCFRSolver::ApplyEpsilonRegretMatching(CFRInfoStateValues* isvals) {
  isvals->ApplyRegretMatching();

  // Now mix back in epsilon of the uniform.
  for (int aidx = 0; aidx < isvals->num_actions(); ++aidx) {
    isvals->current_policy[aidx] =
        epsilon_ * 1.0 / isvals->legal_actions.size() +
        (1.0 - epsilon_) * isvals->current_policy[aidx];
  }
}

void EpsilonCFRSolver::ApplyEpsilonRegretMatching() {
  for (auto& key_and_info_state : info_states_) {
    ApplyEpsilonRegretMatching(&key_and_info_state.second);
  }
}

BRInfo NashConvWithEps(const Game& game, const Policy& policy) {
  GameType game_type = game.GetType();
  if (game_type.dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError("The game must be turn-based.");
  }

  std::unique_ptr<State> root = game.NewInitialState();
  absl::flat_hash_map<std::string, std::vector<double>> state_values;

  BRInfo br_info;
  br_info.on_policy_values =
      ExpectedReturns(*root, policy, -1, false, 0, &state_values);
  br_info.deviation_incentives.resize(game.NumPlayers());
  br_info.cvtables.reserve(game.NumPlayers());

  std::vector<double> best_response_values(game.NumPlayers());
  for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
    TabularBestResponse best_response(game, p, &policy, -1.0, &policy, &state_values);
    best_response_values[p] = best_response.Value(*root);
    br_info.cvtables.push_back(best_response.cvtable());
  }

  SPIEL_CHECK_EQ(best_response_values.size(), br_info.on_policy_values.size());
  br_info.nash_conv = 0;
  for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
    br_info.deviation_incentives[p] =
        best_response_values[p] - br_info.on_policy_values[p];
    if (br_info.deviation_incentives[p] <
        -FloatingPointDefaultThresholdRatio()) {
      SpielFatalError(
          absl::StrCat("Negative Nash deviation incentive for player ", p, ": ",
                       br_info.deviation_incentives[p],
                       ". Does you game have imperfect ",
                       "recall, or does State::ToString() not distinguish ",
                       "between unique states?"));
    }
    br_info.nash_conv += br_info.deviation_incentives[p];
  }

  return br_info;
}

}  // namespace algorithms
}  // namespace open_spiel
