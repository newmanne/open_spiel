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

// class Bidder {
//   public:

//     virtual const double Valuation(std::vector<int> package);

// };

// class BasicBidder : Bidder {
//   public:

//     double valuation(std::vector<int> const &package) {
//       // Additive valuations
//       return DotProduct(package, values_);
//     }

//   private:
//     std::vector<double> values_;

// };


class AuctionState : public SimMoveState {
 public:
  explicit AuctionState(std::shared_ptr<const Game> game, 
    int num_players, 
    int max_rounds_,
    std::vector<int> num_licenses, 
    double increment, 
    std::vector<double> open_price, 
    std::vector<int> product_activity,
    int undersell_rule,
    int information_policy,
    bool allow_negative_profit_bids,
    bool tiebreaks,
    std::vector<std::vector<std::vector<double>>> values,
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
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  std::vector<std::string> ToHidden(const std::vector<int>& demand) const;
  std::vector<int> ActionToBid(Action action) const;
  void PostProcess();
  void ProcessBids(const std::vector<std::vector<Player>> player_order);
  void ChanceOutcomeToOrdering();
  bool DetermineTiebreaks();
  void CheckBid(const std::vector<int>& bid) const;

 
  // Initialized to invalid values. Use Game::NewInitialState().
  Player cur_player_;  // Player whose turn it is.

  // Param info
  int num_players_;
  int num_products_;
  int round_;
  std::vector<int> num_licenses_;
  double increment_;
  std::vector<double> open_price_;
  std::vector<int> product_activity_;

  std::vector<int> final_bids_;

  int undersell_rule_;
  int information_policy_;
  bool allow_negative_profit_bids_;
  bool tiebreaks_;

  // Type info
  std::vector<std::vector<std::vector<double>>> values_;
  std::vector<std::vector<double>> budgets_;
  std::vector<std::vector<double>> type_probs_;

  // Used to encode the information state.

  // What the bidder submits: Player X Round X Product
  std::vector<std::vector<std::vector<int>>> submitted_demand_;
  // What the bidder is allocated: Player X Round X Product
  std::vector<std::vector<std::vector<int>>> processed_demand_;

  // Prices by round
  std::vector<std::vector<double>> sor_price_;
  std::vector<std::vector<double>> clock_price_;
  std::vector<std::vector<double>> posted_price_;

  // Value by player by product
  std::vector<std::vector<double>> value_;
  // Budget by player
  std::vector<double> budget_;
  // Activity by player
  std::vector<int> activity_;

  // Processed aggregate demand for each product
  std::vector<std::vector<int>> aggregate_demands_;

  // Mapping from ActionID -> Bid
  std::vector<std::vector<int>> all_bids_;
  // Mapping from ActionID -> Activity
  std::vector<int> all_bids_activity_;

  std::vector<std::vector<Player>> default_player_order_;
  std::vector<std::vector<Player>> tie_breaks_needed_;
  std::vector<std::vector<Player>> selected_order_;
  int tie_break_index_; // What product are we currently on a chance node for?

  bool finished_;

  // Used for tensor
  int max_rounds_;
};

class AuctionGame : public SimMoveGame {
 public:
  explicit AuctionGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -max_budget_; } // Not an exact calculation, but a lower bound
  double MaxUtility() const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  int SizeHelper(int max_rounds) const;

 private:
  int num_players_;

  // Number of products
  int num_products_;
  // Number of licenses per product
  std::vector<int> num_licenses_;
  // Clock increment
  double increment_;
  // Opening prices for each product
  std::vector<double> open_price_;
  // Activity for each product
  std::vector<int> product_activity_;

  // Type information
  std::vector<std::vector<std::vector<double>>> values_;
  std::vector<std::vector<double>> budgets_;
  std::vector<std::vector<double>> type_probs_;
  

  int undersell_rule_;
  int information_policy_;
  bool allow_negative_profit_bids_;
  bool tiebreaks_;

  int max_chance_outcomes_;
  double max_value_;
  double max_budget_;

    // Used for tensor
  int max_rounds_;
};

}  // namespace clock_auction
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CLOCK_AUCTION_H
