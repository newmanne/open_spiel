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

#ifndef OPEN_SPIEL_GAMES_AUCTION_H
#define OPEN_SPIEL_GAMES_AUCTION_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

//
// Parameters:
//   "players"     int    number of players                      (default = 2)

namespace open_spiel {
namespace auction {

class AuctionGame;

class AuctionState : public State {
 public:
  explicit AuctionState(std::shared_ptr<const Game> game, int num_licenses, double increment, double open_price);
  // AuctionState(const AuctionState&) = default;

  void Reset(const GameParameters& params);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(
      Player player, std::vector<double>* values) const override;
  void ObservationTensor(
      Player player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  // Initialized to invalid values. Use Game::NewInitialState().
  Player cur_player_;  // Player whose turn it is.
  int total_moves_;
  int player_moves_;

  // Used to encode the information state.
  std::vector<std::vector<int>> bidseq_;
  std::string bidseq_str_;
  std::vector<double> price_;
  std::vector<double> value_;
  std::vector<double> budget_;
  bool underdemand_;
  
  int num_licenses_;

  double increment_;

  // std::vector<double> value_types_;
  // std::vector<double> budget_types_;

};

class AuctionGame : public Game {
 public:
  explicit AuctionGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  // TODO: Does this mean we should consider compressing our ranges?
  double MinUtility() const override { return -num_licenses_ * open_price_ * 10; } // TODO: Can't this be infinitely bad? Or depend on max game length and increments if you are forced to buy all the licenses at the lowest value?
  double MaxUtility() const override { return (200 - open_price_) * num_licenses_; } // TODO: Highest value at opening price for all items?
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new AuctionGame(*this));
  }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;

  int NumLicenses() const { return num_licenses_; }

 private:
  // Number of players.
  int num_players_;

  // Total licenses in the game, determines the legal bids.
  int num_licenses_;

  double increment_;
  double open_price_;

  // std::vector<double> value_types_;
  // std::vector<double> budget_types_;
};

}  // namespace auction
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_AUCTION_H
