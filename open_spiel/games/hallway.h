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

#ifndef OPEN_SPIEL_GAMES_HALLWAY_H_
#define OPEN_SPIEL_GAMES_HALLWAY_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace hallway {

// Constants.
inline constexpr int kNumPlayers = 1;
inline constexpr int kNumActions = 4;  // Right, Left

inline constexpr int kDefaultHeight = 1;
inline constexpr int kDefaultWidth = 20;
inline constexpr int kDefaultHorizon = 100;

class HallwayGame;

// State of an in-play game.
class HallwayState : public State {
 public:
  HallwayState(std::shared_ptr<const Game> game);
  HallwayState(const HallwayState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move) override;

 private:
  // Check if player position is in bottom row between start and goal.
  bool IsCliff(int row, int col) const;

  double IsGoal(int row, int col) const;

  int height_;
  int width_;
  int horizon_;

  int player_row_;
  int player_col_; // Starting point
  int time_counter_ = 0;
};

// Game object.
class HallwayGame : public Game {
 public:
  explicit HallwayGame(const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new HallwayState(shared_from_this()));
  }
  std::vector<int> ObservationTensorShape() const override {
    return {height_, width_};
  }
  std::vector<int> InformationStateTensorShape() const override {
    return {NumDistinctActions() * horizon_};
  }

  int NumDistinctActions() const override { 
    return kNumActions;
    // if (height_ > 1 && width_ > 1) {
    //   return 4;
    // } else {
    //   return 2;
    // }
  }

  int NumPlayers() const override { return kNumPlayers; }
  double MaxUtility() const override { return 5; }
  double MinUtility() const override { return 0; }
  int MaxGameLength() const override { return horizon_; }
  int Height() const { return height_; }
  int Width() const { return width_; }

 private:
  const int height_;
  const int width_;
  const int horizon_;
};

}  // namespace hallway
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_HALLWAY_H_