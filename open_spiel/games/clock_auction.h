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

namespace open_spiel {
namespace clock_auction {


int Sum(std::vector<int> v) {
  return std::accumulate(v.begin(), v.end(), 0);
}

double Sum(std::vector<double> v) {
  return std::accumulate(v.begin(), v.end(), 0.);
}

int DotProduct(std::vector<int> const &a, std::vector<int> const &b) {
  return std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.0);
}

double DotProduct(std::vector<int> const &a, std::vector<double> const &b) {
  return std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.0);
}

int Factorial(int x) {
  return (x == 0) || (x == 1) ? 1 : x * Factorial(x-1);
}

// See https://www.geeksforgeeks.org/how-to-find-index-of-a-given-element-in-a-vector-in-cpp/
template <typename T>
int FindIndex(std::vector<T> v, T K) {
    auto it = std::find(v.begin(), v.end(), K);
 
    // If element was found
    if (it != v.end()) {
        // calculating the index
        int index = it - v.begin();
        return index;
    }
    SpielFatalError("Could not find bid?");
}


class Bidder {
  public:
    virtual const double ValuationForPackage(std::vector<int> const &package) const = 0;
    virtual const double GetBudget() const = 0;
    virtual const double GetPricingBonus() const = 0;
    virtual operator std::string() const = 0;
};

class LinearBidder : public Bidder {

  public:
    explicit LinearBidder(std::vector<double> values, double budget, double pricing_bonus) : values_(values), budget_(budget), pricing_bonus_(pricing_bonus) {
    }

    const double ValuationForPackage(std::vector<int> const &package) const override {
      return DotProduct(package, values_);
    }

    const double GetBudget() const override {
      return budget_;
    }

    const double GetPricingBonus() const override {
      return pricing_bonus_;
    }

    operator std::string() const {
        return absl::StrCat("LinearValues:", absl::StrJoin(values_, ", "), " Budget: ", budget_);
    }

    friend std::ostream& operator<<(std::ostream& os, const LinearBidder& b);


  private:
    double budget_;
    std::vector<double> values_;
    double pricing_bonus_;

};

class MarginalBidder : public Bidder {

  public:
    explicit MarginalBidder(std::vector<std::vector<double>> values, double budget, double pricing_bonus) : values_(values), budget_(budget), pricing_bonus_(pricing_bonus) {
    }

    const double ValuationForPackage(std::vector<int> const &package) const override {
      double value = 0.;
      for (int i = 0; i < package.size(); i++) {
        int quantity = package[i];
        for (int j = 0; j < quantity; j++) {
          value += values_[i][j];
        }
      }
      return value;
    }

    const double GetBudget() const override {
      return budget_;
    }

    const double GetPricingBonus() const override {
      return pricing_bonus_;
    }

    operator std::string() const {
        std::string value_string = "";
        for (int i = 0; i < values_.size(); i++) {
          absl::StrAppend(&value_string, absl::StrJoin(values_[i], ", "));
        }
        return absl::StrCat("MarginalValues:", value_string, " Budget: ", budget_);
    }

    friend std::ostream& operator<<(std::ostream& os, const LinearBidder& b);


  private:
    double budget_;
    std::vector<std::vector<double>> values_;
    double pricing_bonus_;

};

class EnumeratedValueBidder : public Bidder {

  public:
    explicit EnumeratedValueBidder(std::vector<double> values, double budget, double pricing_bonus, std::vector<std::vector<int>> all_bids) : values_(values), budget_(budget), pricing_bonus_(pricing_bonus), all_bids_(all_bids) {
    }

    const double ValuationForPackage(std::vector<int> const &package) const override {
      int index = FindIndex(all_bids_, package);
      return values_[index];
    }

    const double GetBudget() const override {
      return budget_;
    }

    const double GetPricingBonus() const override {
      return pricing_bonus_;
    }

    operator std::string() const {
        // TODO: Probably stupidly long
        std::string value_string = absl::StrJoin(values_, ", ");
        return absl::StrCat("Values:", value_string, " Budget: ", budget_);
    }

    friend std::ostream& operator<<(std::ostream& os, const LinearBidder& b);


  private:
    double budget_;
    std::vector<double> values_;
    double pricing_bonus_;
    std::vector<std::vector<int>> all_bids_;
};


std::ostream& operator<<(std::ostream &strm, const LinearBidder &a) {
  return strm << std::string(a);
}

class AuctionGame;

class AuctionState : public SimMoveState {
 public:
  explicit AuctionState(std::shared_ptr<const Game> game, 
    int num_players, 
    int max_rounds_,
    std::vector<int> num_licenses, 
    double increment, 
    std::vector<double> open_price, 
    std::vector<int> product_activity,
    std::vector<int> all_bids_activity,
    std::vector<std::vector<int>> all_bids,
    int undersell_rule,
    int information_policy,
    bool activity_on,
    bool allow_negative_profit_bids,
    bool tiebreaks,
    double switch_penalty_,
    std::vector<std::vector<Bidder*>>,
    std::vector<std::vector<double>> type_probs,
    int max_n_types_
  );

  void Reset(const GameParameters& params);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player, absl::Span<float> values) const override;
  int InformationStateTensorSize() const override;

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
  bool activity_on_;
  bool allow_negative_profit_bids_;
  bool tiebreaks_;

  // Type info
  std::vector<std::vector<Bidder*>> bidders_;
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

  // Bidder realization
  std::vector<Bidder*> bidder_;
  std::vector<int> types_;
  int max_n_types_;

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


  std::vector<int> num_switches_;
  double switch_penalty_;

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
  std::vector<std::vector<Bidder*>> bidders_;
  std::vector<std::vector<double>> type_probs_;
  int max_n_types_;

  int undersell_rule_;
  bool activity_on_;
  int information_policy_;
  bool allow_negative_profit_bids_;
  bool tiebreaks_;

  int max_chance_outcomes_;
  double max_value_;
  double max_budget_;

    // Used for tensor
  int max_rounds_;

  double switch_penalty_;

  // Mapping from ActionID -> Bid
  std::vector<std::vector<int>> all_bids_;
  // Mapping from ActionID -> Activity
  std::vector<int> all_bids_activity_;

};

}  // namespace clock_auction
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CLOCK_AUCTION_H
