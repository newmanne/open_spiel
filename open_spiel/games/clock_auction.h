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

#ifndef OPEN_SPIEL_GAMES_CLOCK_AUCTION_H
#define OPEN_SPIEL_GAMES_CLOCK_AUCTION_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

//
// Parameters:
//   "players"     int    number of players                      (default = 2)

namespace open_spiel {
namespace clock_auction {

class AuctionGame;

class AuctionState : public SimMoveState {
 public:
  explicit AuctionState(std::shared_ptr<const Game> game, 
    int num_players, 
    int num_licenses, 
    double increment, 
    double open_price, 
    int undersell_rule_,
    int infomration_policy_,
    std::vector<std::vector<double>> values,
    std::vector<std::vector<double>> budgets,
    std::vector<std::vector<double>> type_probs
  );

  void Reset(const GameParameters& params);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player, absl::Span<float> values) const override;
  void ObservationTensor(Player player, absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions);

 private:
  std::vector<Player> PlayersThatWantToDrop() const;
  std::vector<int> RequestedDrops() const;
  void HandleUndersell(Action action);
 
  // Initialized to invalid values. Use Game::NewInitialState().
  Player cur_player_;  // Player whose turn it is.
  int total_moves_;
  int player_moves_;

  // Param info
  int num_players_;
  int num_licenses_;
  double increment_;
  double open_price_;
  int undersell_rule_;
  int information_policy_;
  std::vector<std::vector<double>> values_;
  std::vector<std::vector<double>> budgets_;
  std::vector<std::vector<double>> type_probs_;

  // Used to encode the information state.
  std::vector<std::vector<int>> bidseq_;
  std::vector<int> final_bids_;
  std::vector<Player> undersell_order_;
  std::vector<double> price_;
  std::vector<double> value_;
  std::vector<double> budget_;
  std::vector<int> aggregate_demands_;
  
  bool finished_;
  bool undersell_;
};

class AuctionGame : public Game {
 public:
  explicit AuctionGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -max_budget_; } // Not an exact calculation, but a lower bound
  double MaxUtility() const override { return (max_value_ - open_price_) * num_licenses_; } // Winning all licenses at the opening price with the highest value. This isn't a perfect computation because budgets might constrain you
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;

 private:
  int num_players_;
  int num_licenses_;
  double increment_;
  double open_price_;
  int undersell_rule_;
  int information_policy_;
  std::vector<std::vector<double>> values_;
  std::vector<std::vector<double>> budgets_;
  std::vector<std::vector<double>> type_probs_;

  int max_chance_outcomes_;
  double max_value_;
  double max_budget_;

};

}  // namespace clock_auction
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CLOCK_AUCTION_H
