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
#include <cstdlib>
#include <cstring>
#include <fstream>


#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/file.h"


namespace open_spiel {
namespace clock_auction {

namespace {

// Default Parameters.
constexpr int kMaxPlayers = 10;
constexpr int kMoveLimit = 100;

// Facts about the game
const GameType kGameType{/*short_name=*/"clock_auction",
                         /*long_name=*/"Clock Auction",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/kMaxPlayers,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/false,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                          {
                           {"filename", GameParameter(GameParameter::Type::kString)},
                          }
                        };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AuctionGame(params));
}

}  // namespace

REGISTER_SPIEL_GAME(kGameType, Factory);

AuctionState::AuctionState(std::shared_ptr<const Game> game, 
  int num_players,
  int num_licenses,
  double increment,
  double open_price,
  bool undersell_rule_,
  std::vector<std::vector<double>> values,
  std::vector<std::vector<double>> budgets,
  std::vector<std::vector<double>> probs
) : 
      SimMoveState(game),
      cur_player_(kChancePlayerId), // Chance begins by setting types
      total_moves_(0),
      player_moves_(0),
      undersell_(false),
      finished_(false),
      num_licenses_(num_licenses),
      num_players_(num_players),
      increment_(increment),
      open_price_(open_price),
      undersell_rule_(undersell_rule_),
      values_(values),
      budgets_(budgets),
      type_probs_(probs),
      bidseq_(),
      aggregate_demands_(),
      undersell_order_(),
      final_bids_() {
      for (auto p = Player{0}; p < num_players_; p++) {
        std::vector<int> demand;
        bidseq_.push_back(demand);
      }
      price_.push_back(open_price_);
}

std::vector<int> AuctionState::RequestedDrops() const {
    std::vector<int> requested_drops; // Track of how many drops each player wants
    for (auto p = Player{0}; p < num_players_; ++p) {
      requested_drops.push_back(final_bids_[p] - bidseq_[p].back());
    }
    return requested_drops;
}


std::vector<Player> AuctionState::PlayersThatWantToDrop() const {
  // Get the players that still want to drop
  std::vector<int> requested_drops = this->RequestedDrops();
  std::vector<Player> droppers;
  for (int i = 0; i < requested_drops.size(); i++) {
    if (requested_drops[i] > 0) {
      droppers.push_back(i);
    }
  }
  return droppers;
}

int Sum(std::vector<int> v) {
  int s = 0;
  for (auto& e : v) {
    s += e;
  }
  return s;
}

void AuctionState::DoApplyActions(const std::vector<Action>& actions) {
  // Check the actions are valid.
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  for (auto p = Player{0}; p < num_players_; ++p) {
    const int action = actions[p];
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LE(action, bidseq_[p].empty() ? num_licenses_ : bidseq_[p].back());
    bidseq_[p].push_back(action);
    player_moves_++;
  }

  int aggregateDemand = 0;
  for (auto p = Player{0}; p < num_players_; ++p) {
    aggregateDemand += bidseq_[p].back();
  }
  aggregate_demands_.push_back(aggregateDemand);

  if (aggregateDemand > num_licenses_) {
    // Normal case: Increment price, since demand > supply
    double next_price = price_.back() * (1 + increment_);
    price_.push_back(next_price);
  } else {
    if (aggregateDemand == num_licenses_ || (aggregateDemand <= num_licenses_ && (!undersell_rule_ || price_.size() == 1))) {
      // Demand <= supply. We are finished.
      // Undersell in the first round is always allowed without penalty, so we won't set the flag
      for (auto p = Player{0}; p < num_players_; ++p) {
        final_bids_.push_back(bidseq_[p].back());
      }
      finished_ = true;
    } else {
      SPIEL_CHECK_TRUE(undersell_rule_);
      for (auto p = Player{0}; p < num_players_; ++p) {
        final_bids_.push_back(bidseq_[p].end()[-2]); // Use prev round bids and we'll drop according to chance
      }
      undersell_ = true;
      cur_player_ = kChancePlayerId; // At least someone wanted to drop. Chance will decide who gets to drop
    }
  }
}

std::string AuctionState::ActionToString(Player player, Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  if (player != kChancePlayerId) {
    return absl::StrCat("Bid for ", action_id, " licenses @ ", price_.back());
  } else {
    if (value_.size() < num_players_) {
      return absl::StrCat("Player ", value_.size(), " was assigned a value of ", values_[value_.size()][action_id], " and a budget of ", budgets_[budget_.size()][action_id]);
    } else if (undersell_) {
      return absl::StrCat("Undersell! Allowing player ", action_id, " to drop");
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
  if (undersell_) { // Chance node handles undersell
    HandleUndersell(action);
    return;
  }

  if (value_.size() < num_players_) { // Chance node assigns a value and budget to a player
    auto player_index = value_.size();
    SPIEL_CHECK_GE(player_index, 0);
    value_.push_back(values_[player_index][action]); 
    budget_.push_back(budgets_[player_index][action]);
  } 

  if (value_.size() == num_players_) { // All of the assignments have been made
    cur_player_ = kSimultaneousPlayerId;
  }
}

void AuctionState::HandleUndersell(Action action) {
  undersell_order_.push_back(action);

  // Compute allowed number of drops
  int required_drops = Sum(final_bids_) - num_licenses_;
  int requested_drops = RequestedDrops()[action];
  SPIEL_CHECK_GE(required_drops, 1);
  SPIEL_CHECK_GE(requested_drops, 1);

  while (required_drops > 0 && requested_drops > 0) {
    required_drops--;
    requested_drops--;
    final_bids_[action]--;
  }

  if (required_drops == 0) { // If we're done dropping
    finished_ = true;
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

  int limit = bidseq_[player].empty() ? num_licenses_ : bidseq_[player].back();
  double price = price_.back();
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

  if (undersell_) { // Chance node is for undersell
      SPIEL_CHECK_TRUE(undersell_rule_);
      std::vector<Player> want_to_drop = PlayersThatWantToDrop();
      SPIEL_CHECK_FALSE(want_to_drop.empty());
      for (auto& player : want_to_drop) {
        valuesAndProbs.push_back({player, 1. / want_to_drop.size()});
      }
      return valuesAndProbs;
  }

  std::vector<double> probs;
  if (value_.size() < num_players_) { // Chance node is assigning a type
    auto player_index = value_.size();
    probs = type_probs_[player_index];
  } 

  for(int i = 0; i < probs.size(); i++) {
    valuesAndProbs.push_back({i, probs[i]});
  }
  return valuesAndProbs;
}

std::string AuctionState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string result = absl::StrCat("p", player);
  if (value_.size() > player) {
    absl::StrAppend(&result, absl::StrCat("v", value_[player], "b", budget_[player]));  
  }
  if (!bidseq_[player].empty()) {
    absl::StrAppend(&result, absl::StrCat("\n", absl::StrJoin(bidseq_[player], ","), "\n"));
    absl::StrAppend(&result, absl::StrJoin(aggregate_demands_, ","));
  }
  return result;
}

std::string AuctionState::ToString() const {
  std::string result = "";
  // Player type storage
  for (auto p = Player{0}; p < num_players_; p++) {
      if (value_.size() > p) {
        absl::StrAppend(&result, absl::StrCat("Player(n=", p, ", Value=", value_[p], ", Budget=", budget_[p], ")\n"));  
      }
  }

  absl::StrAppend(&result, absl::StrCat("Price: ", price_.back(), "\n"));

  for (auto p = Player{0}; p < num_players_; p++) {
    if (!bidseq_[p].empty()) {
      absl::StrAppend(&result, absl::StrCat("Player ", p, " demanded: ", absl::StrJoin(bidseq_[p], " "), "\n"));
    }
  }
  if (!aggregate_demands_.empty()) {
    absl::StrAppend(&result, absl::StrCat("Aggregate demands: ", absl::StrJoin(aggregate_demands_, " ")));
  }
  if (!final_bids_.empty()) {
    absl::StrAppend(&result, absl::StrCat("\nFinal bids: ", absl::StrJoin(final_bids_, " ")));
  }
  if (undersell_) {
    absl::StrAppend(&result, "\nUndersell");
    if (!undersell_order_.empty()) {
      absl::StrAppend(&result, absl::StrCat("\nUndersell order: ", absl::StrJoin(undersell_order_, " ")));
    } 
  }

  return result;
}

bool AuctionState::IsTerminal() const { 
  if (player_moves_ >= kMoveLimit) {
    std::cerr << "Number of player moves exceeded move limit of " << kMoveLimit << "! Terminating prematurely...\n" << std::endl;
  }
  return finished_ || player_moves_ >= kMoveLimit; 
}

std::vector<double> AuctionState::Returns() const {
  int final_demand = Sum(final_bids_);
  if (undersell_) {
    SPIEL_CHECK_EQ(final_demand, num_licenses_);
  } else {
    SPIEL_CHECK_LE(final_demand, num_licenses_);
  }

  std::vector<double> returns(num_players_, 0.0);
  double price = undersell_ ? price_.end()[-2] : price_.back(); // If the undersell flag is triggered, let's assume the drop bids occured at start-of-round, and use the previous round's price. If not using the undersell_rule_, the more intuitive thing happens nad we just use the price

  for (auto p = Player{0}; p < num_players_; p++) {
    SPIEL_CHECK_GE(final_bids_[p], 0);
    returns[p] = (value_[p] - price) * final_bids_[p];
  }
  return returns;
}

std::unique_ptr<State> AuctionState::Clone() const {
  return std::unique_ptr<State>(new AuctionState(*this));
}

// Annoyingly hyper-sensitive to double vs int in JSON lib
double ParseDouble(json::Value val) {
  return val.IsDouble() ? val.GetDouble() : val.GetInt();
}

void CheckRequiredKey(json::Object obj, std::string key) {
  if (obj.find(key) == obj.end()) {
    SpielFatalError(absl::StrCat("Missing JSON key: ", key));
  }
}

AuctionGame::AuctionGame(const GameParameters& params) :
   Game(kGameType, params) {

  std::string filename;
  if (IsParameterSpecified(game_parameters_, "filename")) {
    filename = ParameterValue<std::string>("filename");
  } else {
    std::cerr << "No file input specified. Using defaults" << std::endl;
    filename = "parameters.json";
  }

  std::cerr << "Parsing configuration from " << filename << std::endl;
  std::string string_data = file::ReadContentsFromFile(filename, "r");
  SPIEL_CHECK_GT(string_data.size(), 0);

  absl::optional<json::Value> v = json::FromString(string_data);
  auto object = v->GetObject();

  CheckRequiredKey(object, "players");
  auto players = object["players"].GetArray();
  num_players_ = players.size();
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);

  CheckRequiredKey(object, "opening_price");
  open_price_ = ParseDouble(object["opening_price"]);
  CheckRequiredKey(object, "licenses");
  num_licenses_ = object["licenses"].GetInt();
  CheckRequiredKey(object, "increment");
  increment_ = ParseDouble(object["increment"]);
  CheckRequiredKey(object, "undersell_rule");
  undersell_rule_ = object["undersell_rule"].GetBool();

  // Loop over players, parsing values and budgets
  max_value_ = 0.;
  max_budget_ = 0.;
  for (auto p = Player{0}; p < num_players_; p++) {
    auto player_object = players[p].GetObject();
    CheckRequiredKey(player_object, "type");
    auto type_array = player_object["type"].GetArray();
    std::vector<double> player_budgets;
    std::vector<double> player_values;
    std::vector<double> player_probs;
    for (auto t = 0; t < type_array.size(); t++) {
      auto type_object = type_array[t].GetObject();
      CheckRequiredKey(type_object, "value");
      CheckRequiredKey(type_object, "budget");
      CheckRequiredKey(type_object, "prob");
      double value = ParseDouble(type_object["value"]);
      if (value > max_value_) {
        max_value_ = value;
      }
      player_values.push_back(value);
      double budget = ParseDouble(type_object["budget"]);
      if (budget > max_budget_) {
        max_budget_ = budget;
      }
      player_budgets.push_back(budget);
      double prob = ParseDouble(type_object["prob"]);
      player_probs.push_back(prob);
    }
    values_.push_back(player_values);
    budgets_.push_back(player_budgets);
    type_probs_.push_back(player_probs);
  }
  
  // Compute max chance outcomes
  // Max of # of type draws and tie-breaking
  std::vector<int> lengths;
  lengths.push_back(num_players_);
  for (auto p = Player{0}; p < num_players_; p++) {
    lengths.push_back(type_probs_[p].size());
  }
  max_chance_outcomes_ = *std::max_element(lengths.begin(), lengths.end());
}

int AuctionGame::NumDistinctActions() const {
  return num_licenses_ + 1; // Bid for any number of licenes, including 0
}

std::unique_ptr<State> AuctionGame::NewInitialState() const {
  std::unique_ptr<AuctionState> state(
      new AuctionState(shared_from_this(), num_players_, num_licenses_, increment_, open_price_, undersell_rule_, values_,  budgets_, type_probs_));
  return state;
}

int AuctionGame::MaxChanceOutcomes() const { 
  return max_chance_outcomes_;
}

int AuctionGame::MaxGameLength() const {
  return kMoveLimit; // In theory, this game is bounded only by the budgets and can go on much longer
}

std::vector<int> AuctionGame::ObservationTensorShape() const {
  SpielFatalError("Unimplemented ObservationTensorShape");  
}

void AuctionState::ObservationTensor(Player player, absl::Span<float> values) const {
  SpielFatalError("Unimplemented ObservationTensor");
}

std::vector<int> AuctionGame::InformationStateTensorShape() const {
  SpielFatalError("Unimplemented InformationStateTensorShape");
}

void AuctionState::InformationStateTensor(Player player, absl::Span<float> values) const {
  SpielFatalError("Unimplemented InformationStateTensor");
}

}  // namespace clock_auction
}  // namespace open_spiel
