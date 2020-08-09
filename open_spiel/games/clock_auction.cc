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

#include "open_spiel/games/clock_auction.h"

#include <algorithm>
#include <array>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace clock_auction {

namespace {
// Default Parameters.
constexpr int kDefaultPlayers = 2;
// Number of licenses available for sale
constexpr int kLicenses = 3;  
constexpr int kInvalidOutcome = -1;
constexpr int kInvalidBid = -1;

// How much to multiply the price of a license by in each round
constexpr double defaultIncrement = 0.1;
// The starting price of a license
constexpr double defaultOpenPrice = 100.0;

// Type space. Assuming linear values for now, so your value for licenses is v * x, where x is the number of licenses you win
std::vector<int> values {125, 150};
// Budgets. You cannot bid above your budget.
std::vector<int> budgets {350, 400};

constexpr double move_limit = 25;

// Facts about the game
const GameType kGameType{/*short_name=*/"clock_auction",
                         /*long_name=*/"Clock Auction",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/kDefaultPlayers,
                         /*min_num_players=*/kDefaultPlayers,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/false,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                          {
                           {"players", GameParameter(kDefaultPlayers)},
                           {"num_licenses", GameParameter(kLicenses)},
                           {"open_price", GameParameter(defaultOpenPrice)},
                           {"increment", GameParameter(defaultIncrement)},
                          //  {"budget_types", GameParameter()},
                          //  {"value_types", GameParameter()},
                          }
                        };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AuctionGame(params));
}
}  // namespace

REGISTER_SPIEL_GAME(kGameType, Factory);

AuctionState::AuctionState(std::shared_ptr<const Game> game, int num_licenses, double increment, double open_price)
    : SimMoveState(game),
      cur_player_(kChancePlayerId), // Chance begins by setting types
      total_moves_(0),
      player_moves_(0),
      undersell_(false),
      finished_(false),
      num_licenses_(num_licenses),
      increment_(increment),
      bidseq_(),
      aggregate_demands_(),
      bidseq_str_() {
      for (auto p = Player{0}; p < num_players_; p++) {
        std::vector<int> demand;
        bidseq_.push_back(demand);
      }
      price_.push_back(open_price);
}

void AuctionState::DoApplyActions(const std::vector<Action>& actions) {
  // Check the actions are valid.
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  int aggregateDemand = 0;
  for (auto p = Player{0}; p < num_players_; ++p) {
    const int action = actions[p];
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LE(action, bidseq_[p].empty() ? num_licenses_ : bidseq_[p].back());
    bidseq_[p].push_back(action);
    aggregateDemand += action;
    player_moves_++;
  }

  aggregate_demands_.push_back(aggregateDemand);

  bool undersell = aggregateDemand < num_licenses_;

  if ((undersell && price_.size() == 1) || aggregateDemand == num_licenses_) {
    // Undersell in the first round is allowed without penalty
    finished_ = true;
  } else if (undersell) {
      finished_ = true;
      // TODO: Think harder about what you want to do with undersell
      // undersell_ = true;
      // cur_player_ = kChancePlayerId;  
  } else {
    // Increment price
    price_.push_back(price_.back() * (1 + increment_));
  }
}

std::string AuctionState::ActionToString(Player player, Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  if (player != kChancePlayerId) {
    return absl::StrCat("Player ", player, " bid for ", action_id, " licenses");
  } else {
    if (value_.size() < num_players_) {
      return absl::StrCat("Player ", value_.size(), " was assigned a value of ", values[action_id]);
    } else if (budget_.size() < num_players_) {
      return absl::StrCat("Player ", budget_.size(), " was assigned a budget of ", budgets[action_id]);
    } else if (undersell_) {
      return absl::StrCat("Assigning undersell drop to ", action_id);
    }
  }
}

int AuctionState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

void AuctionState::DoApplyAction(Action action) {
  total_moves_++;

  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action);
    return;
  }
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (value_.size() < num_players_) {
    value_.push_back(values[action]); 
  } else if (budget_.size() < num_players_) {
    budget_.push_back(budgets[action]);
  } else if (undersell_) {
      SpielFatalError("Unimplemented undersell");  
    // // If there is undersell and it is not the first round, we let one player drop randomly so that the entire supply of units are sold. This is a bit strange, but the best I can think of without enhancing the action space to include intra-round bidding
    // // int prevDemands = aggregate_demands.end()[-2];
    // int allowedDrops = num_licenses_ - aggregate_demands_.back();
    // bidseq_[action].push_back(bidseq_[action].back() - allowedDrops);
    // SPIEL_CHECK_GE(bidseq_[action].back(), 0);

    // int aggregateDemand = 0;
    // for (auto p = Player{0}; p < num_players_; ++p) {
    //   aggregateDemand += bidseq_[p].back();
    // }
    // SPIEL_CHECK_EQ(aggregateDemand, num_licenses_);
    // finished_ = true;
  }

  if (budget_.size() == num_players_) {
    cur_player_ = kSimultaneousPlayerId;
  }
}

std::vector<Action> AuctionState::LegalActions(Player player) const {
  if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
  if (player == kChancePlayerId) return LegalChanceOutcomes();
  if (player == kTerminalPlayerId) return std::vector<Action>();
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Any move weakly lower than a previous bid is allowed if it fits in budget
  std::vector<Action> actions;

  int limit = num_licenses_;
  double price = price_.back();
  if (bidseq_[player].size() > 0) {
    limit = bidseq_[player].back();
  }
  for (int b = 0; b <= limit; b++) {
    if (price * b > budget_[player]) {
      break;
    }
    actions.push_back(b);
  }

  return actions;
}

std::vector<std::pair<Action, double>> AuctionState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  ActionsAndProbs valuesAndProbs;
  valuesAndProbs.push_back(std::make_pair(0, 1. / 2));
  valuesAndProbs.push_back(std::make_pair(1, 1. / 2));
  return valuesAndProbs;
}

std::string AuctionState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string result = absl::StrCat("p", player);
  if (value_.size() > player) {
    absl::StrAppend(&result, absl::StrCat("v", value_[player]));  
  }
  if (budget_.size() > player) {
    absl::StrAppend(&result, absl::StrCat("b", budget_[player]));
  }
  for (int i = 0; i < bidseq_[player].size(); i++) {
    absl::StrAppend(&result, absl::StrCat("r", i + 1, " p", player, " b", bidseq_[player][i], " d", aggregate_demands_[i], "\n"));
  }
  return result;
}

std::string AuctionState::ToString() const {
  // TODO: Write this better
  std::string result = "";
  absl::StrAppend(&result, absl::StrCat("Price: ", price_.back(), "."));

  for (auto p = Player{0}; p < num_players_; p++) {
    if (p != 0) absl::StrAppend(&result, " ");
    absl::StrAppend(&result, absl::StrJoin(bidseq_[p], " "));
  }

  if (IsChanceNode()) {
    return "";
  }
  return result;
}

bool AuctionState::IsTerminal() const { 
  return finished_ || player_moves_ >= move_limit; 
}

std::vector<double> AuctionState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);
  for (auto p = Player{0}; p < num_players_; p++) {
    returns[p] = (value_[p] - price_.back()) * bidseq_[p].back();
  }
  return returns;
}

std::unique_ptr<State> AuctionState::Clone() const {
  return std::unique_ptr<State>(new AuctionState(*this));
}

AuctionGame::AuctionGame(const GameParameters& params)
    : Game(kGameType, params) {
  num_players_ = ParameterValue<int>("players");
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);

  num_licenses_ = ParameterValue<int>("num_licenses");
  increment_ = ParameterValue<double>("increment");
  open_price_ = ParameterValue<double>("open_price");
}

int AuctionGame::NumDistinctActions() const {
  return num_licenses_ + 1; // Bid for any number of licenes, including 0
}

std::unique_ptr<State> AuctionGame::NewInitialState() const {
  std::unique_ptr<AuctionState> state(
      new AuctionState(shared_from_this(), num_licenses_, increment_, open_price_));
  return state;
}

int AuctionGame::MaxChanceOutcomes() const { 
  // Budgets and values can be one of two types each. And the last-round tie-breaking can go one of two ways.
  return 2; 
}

int AuctionGame::MaxGameLength() const {
  return 1000; // TODO: Not a real answer, just a high constant since I don't think algs are using this
}

std::vector<int> AuctionGame::ObservationTensorShape() const {
  SpielFatalError("Unimplemented ObservationTensorShape");  
}

void AuctionState::ObservationTensor(Player player, std:: vector<double>* values) const {
  SpielFatalError("Unimplemented ObservationTensor");
}

std::vector<int> AuctionGame::InformationStateTensorShape() const {
  SpielFatalError("Unimplemented InformationStateTensorShape");
}

void AuctionState::InformationStateTensor(Player player, std::vector<double>* values) const {
  SpielFatalError("Unimplemented InformationStateTensor");
}

}  // namespace clock_auction
}  // namespace open_spiel
