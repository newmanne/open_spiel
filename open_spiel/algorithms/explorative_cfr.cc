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

void EpsilonCFRSolver::EvaluateAndUpdatePolicy() {
  ++iteration_;

  for (int player = 0; player < game_->NumPlayers(); player++) {
    ComputeCounterFactualRegret(*root_state_, player, root_reach_probs_,
                                nullptr);
  }

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

std::pair<double, double> NashConvWithEps(const Game& game,
                                          const Policy& policy) {
  GameType game_type = game.GetType();
  if (game_type.dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError("The game must be turn-based.");
  }

  std::unique_ptr<State> root = game.NewInitialState();
  absl::flat_hash_map<std::string, std::vector<double>> state_values;
  std::vector<double> on_policy_values =
      ExpectedReturns(*root, policy, -1, false, &state_values);

  double max_qv_diff = 0.0;
  std::vector<double> best_response_values(game.NumPlayers());
  for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
    TabularBestResponse best_response(game, p, &policy, &policy, &state_values);
    best_response_values[p] = best_response.Value(*root);
    max_qv_diff = std::max(best_response.max_qv_diff(), max_qv_diff);
  }

  SPIEL_CHECK_EQ(best_response_values.size(), on_policy_values.size());
  double nash_conv = 0;
  for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
    double deviation_incentive = best_response_values[p] - on_policy_values[p];
    if (deviation_incentive < -FloatingPointDefaultThresholdRatio()) {
      SpielFatalError(
          absl::StrCat("Negative Nash deviation incentive for player ", p, ": ",
                       deviation_incentive, ". Does you game have imperfect ",
                       "recall, or does State::ToString() not distinguish ",
                       "between unique states?"));
    }
    nash_conv += deviation_incentive;
  }

  return {nash_conv, max_qv_diff};
}

}  // namespace algorithms
}  // namespace open_spiel
