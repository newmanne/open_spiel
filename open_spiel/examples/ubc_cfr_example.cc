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

#include <string>
#include <random>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"

constexpr int kSeed = 230398247;

ABSL_FLAG(std::string, game_name, "kuhn_poker", "Game to run CFR on.");
ABSL_FLAG(int, num_iters, 1000, "How many iters to run for.");
ABSL_FLAG(int, report_every, 100, "How often to report exploitability.");
ABSL_FLAG(bool, turn_based, false, "Use the LoadGameAsTurnBased flag");

// Example code for using CFR+ to solve Kuhn Poker.
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::shared_ptr<const open_spiel::Game> game;
  std::mt19937 rng(kSeed);


if (absl::GetFlag(FLAGS_turn_based)) {
  game = open_spiel::LoadGameAsTurnBased(absl::GetFlag(FLAGS_game_name));
} else {
  game = open_spiel::LoadGame(absl::GetFlag(FLAGS_game_name));
}

  open_spiel::algorithms::CFRSolver solver(*game);
  std::cerr << "Starting CFR on " << game->GetType().short_name
            << "..." << std::endl;

  for (int i = 0; i < absl::GetFlag(FLAGS_num_iters); ++i) {
    solver.EvaluateAndUpdatePolicy();
    if (i % absl::GetFlag(FLAGS_report_every) == 0 ||
        i == absl::GetFlag(FLAGS_num_iters) - 1) {
      double nash_conv = open_spiel::algorithms::NashConv(
          *game, *solver.AveragePolicy());
      std::cerr << "Iteration " << i << " NashConv=" << nash_conv
                << std::endl;

      // const std::shared_ptr<open_spiel::Policy> average_policy = solver.AveragePolicy();

      // // std::cout << PrintPolicy(average_policy) << std::endl;

      // double nash_conv = open_spiel::algorithms::NashConv(*game, *average_policy, true);
      // std::cout << "Game: " << absl::GetFlag(FLAGS_game_name) << ", iters = " << i
      //           << ", NashConv: " << nash_conv << std::endl;


    }
  }
}