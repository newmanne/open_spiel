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

#ifndef OPEN_SPIEL_ALGORITHMS_EXPLORATIVE_CFR_H_
#define OPEN_SPIEL_ALGORITHMS_EXPLORATIVE_CFR_H_

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

class EpsilonCFRSolver : public CFRSolver {
 public:
  EpsilonCFRSolver(const Game& game, double initial_epsilon)
      : CFRSolver(game), epsilon_(initial_epsilon) {}

  void EvaluateAndUpdatePolicy() override;

  double epsilon() const { return epsilon_; }
  void SetEpsilon(double eps) { epsilon_ = eps; }

 private:
  void ApplyEpsilonRegretMatching(CFRInfoStateValues* isvals);
  void ApplyEpsilonRegretMatching();

  double epsilon_;
};

std::pair<double, double> NashConvWithEps(const Game& game,
                                          const Policy& policy);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_EXPLORATIVE_CFR_H_
