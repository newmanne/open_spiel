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

#include <cmath>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/explorative_cfr.h"
#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/games/efg_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(int, num_iters, 1000000, "How many iters to run for.");
ABSL_FLAG(int, report_every, 100000, "How often to report.");

namespace open_spiel {
namespace algorithms {
namespace {

const char* kSeqEqCounterexampleEFG = R"###(
EFG 2 R "Counterexample by Rich Gibson" { "Player 1" "Player 2" } ""

p "P2" 2 1 "P2" { "l" "r" } 0
  t "l" 1 "Outcome l" { 0.0 0.0 }
  p "P1" 1 1 "P1" { "l" "m" "r" } 0
    t "rl" 2 "Outcome rl" { 0.0 0.0 }
    t "rm" 3 "Outcome rl" { 9.0 -9.0 }
    t "rr" 4 "Outcome rr" { 10.0 -10.0 }
)###";

// From here: http://www.itu.dk/~trbj/papers/seqeqsoda.pdf
const char* kGuessTheAceEFG = R"###(
EFG 2 R "Guess the Ace game from Miltersen & Sorensen '06" { "Player 1" "Player 2" } ""

c "ROOT" 1 "ROOT" { "OOT" 51/52 "ASOT" 1/52 } 0
  p "" 1 1 "P1" { "p" "dnp" } 0
    p "" 2 1 "P2" { "go" "ga" } 0
      t "OOT-p-go" 1 "Outcome OOT-p-go" { -1000.0 1000.0 }
      t "OOT-p-ga" 2 "Outcome OOT-p-ga" { 0.0 0.0 }
    t "OOT-dnp" 3 "Outcome OOT-dnp" { 0.0 0.0 }
  p "" 1 1 "P1" { "p" "dnp" } 0
    p "" 2 1 "P2" { "go" "ga" } 0
      t "ASOT-p-go" 4 "Outcome ASOT-p-go" { 0.0 0.0 }
      t "ASOT-p-ga" 5 "Outcome ASOT-p-ga" { -1000.0 1000.0 }
    t "ASOT-dnp" 6 "Outcome ASOT-dnp" { 0.0 0.0 }
)###";

void RunCFRSeqEqExample() {
  std::shared_ptr<const open_spiel::Game> game =
      // efg_game::LoadEFGGame(kSeqEqCounterexampleEFG);
      // efg_game::LoadEFGGame(kGuessTheAceEFG);
      // LoadGame("kuhn_poker");
      LoadGame("kuhn_poker(players=3)");
      // LoadGame("leduc_poker");
  CFRSolver solver(*game);
  std::cerr << "Starting.. "<< std::endl;

  std::cout << std::endl;
  EpsilonCFRSolver eps_solver(*game, 0.001);

  // Explorative CFR
  int decay_freq = 1000;
  double decay_fac = 0.99;
  for (int i = 0; i < absl::GetFlag(FLAGS_num_iters); ++i) {
    // eps_solver.SetEpsilon(0.1);
    if (i > 0 && i % decay_freq == 0) {
      eps_solver.SetEpsilon(std::max(eps_solver.epsilon() * decay_fac, 1e-6));
    }

    eps_solver.EvaluateAndUpdatePolicy();

    if (i % absl::GetFlag(FLAGS_report_every) == 0 ||
        i == absl::GetFlag(FLAGS_num_iters) - 1) {
      TabularPolicy avg_policy = eps_solver.TabularAveragePolicy();
      BRInfo br_info = NashConvWithEps(*game, avg_policy);
      ConditionalValuesTable merged_table = MergeTables(br_info.cvtables);

      std::cout << "Eps-CFR Iteration " << i
                << " epsilon=" << eps_solver.epsilon()
                << " nash_conv=" << br_info.nash_conv
                << " max_qv_diff=" << merged_table.max_qv_diff()
                << " avg_qv_diff=" << merged_table.avg_qv_diff()
                << std::endl;

      // Print the policy to inspect it.
      // std::cout << avg_policy.ToStringSorted() << std::endl;
    }
  }
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel


int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  open_spiel::algorithms::RunCFRSeqEqExample();
}

