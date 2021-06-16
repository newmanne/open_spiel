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

// Undersell rules
constexpr int kUndersellAllowed = 1;
constexpr int kUndersell = 2;
constexpr int kUndersellPrevClock = 3;

// Information policies
constexpr int kShowDemand = 1;
constexpr int kHideDemand = 2;


int Sum(std::vector<int> v) {
  return std::accumulate(v.begin(), v.end(), 0);
}

// TODO: If I knew C++ I'd do something better than the copy/paste here
int DotProduct(std::vector<int> const &a, std::vector<int> const &b) {
  int dp = 0;
  for (int i = 0; i < a.size(); i++) {
    dp += a[i] * b[i];
  }
  return dp;
}

int DotProduct(std::vector<int> const &a, std::vector<double> const &b) {
  int dp = 0;
  for (int i = 0; i < a.size(); i++) {
    dp += a[i] * b[i];
  }
  return dp;
}

// Cartesian product helper functions
void CartesianRecurse(std::vector<std::vector<int>> &accum, std::vector<int> stack,
    std::vector<std::vector<int>> sequences, int index) {
    std::vector<int> sequence = sequences[index];
    for (int i : sequence)
    {       
        stack.push_back(i);
        if (index == 0)
            accum.push_back(stack);
        else
            CartesianRecurse(accum, stack, sequences, index - 1);
        stack.pop_back();
    }
}

std::vector<std::vector<int>> CartesianProduct(std::vector<std::vector<int>> sequences) {
    std::vector<std::vector<int>> accum;
    std::vector<int> stack;
    if (sequences.size() > 0)
        CartesianRecurse(accum, stack, sequences, sequences.size() - 1);
    return accum;
}

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
  std::vector<int> num_licenses,
  double increment,
  std::vector<double> open_price,
  std::vector<int> product_activity,
  int undersell_rule,
  int information_policy,
  bool allow_negative_profit_bids,
  std::vector<std::vector<std::vector<double>>> values,
  std::vector<std::vector<double>> budgets,
  std::vector<std::vector<double>> probs
) :   SimMoveState(game),
      cur_player_(kChancePlayerId), // Chance begins by setting types
      total_moves_(0),
      player_moves_(0),
      undersell_(false),
      finished_(false),
      num_licenses_(num_licenses),
      num_players_(num_players),
      increment_(increment),
      open_price_(open_price),
      product_activity_(product_activity),
      undersell_rule_(undersell_rule),
      information_policy_(information_policy),
      allow_negative_profit_bids_(allow_negative_profit_bids),
      values_(values),
      budgets_(budgets),
      type_probs_(probs),
      bidseq_(),
      aggregate_demands_(),
      undersell_order_(),
      all_bids_activity_(),
      final_bids_() {

      num_products_ = num_licenses_.size();
      activity_ = std::vector<int>(num_players_, -1);
      for (auto p = Player{0}; p < num_players_; p++) {
        std::vector<std::vector<int>> demand;
        bidseq_.push_back(demand);
      }
      price_.push_back(open_price_);

      // Enumerate bids
      std::vector<std::vector<int>> sequences;
      for (int j = 0; j < num_products_; j++) {
        std::vector<int> sequence;
        for (int k = 0; k <= num_licenses[j]; k++) {
          sequence.push_back(k);
        }
        sequences.push_back(sequence);
      }
      all_bids_ = CartesianProduct(sequences);

      for (auto& bid: all_bids_) {
        all_bids_activity_.push_back(DotProduct(bid, product_activity_));
      }
}

std::vector<int> AuctionState::RequestedDrops() const {
    std::vector<int> requested_drops; // Track of how many drops each player wants
    // for (auto p = Player{0}; p < num_players_; ++p) {
    //   requested_drops.push_back(final_bids_[p] - bidseq_[p].back());
    // }
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



int AuctionState::BidToActivity(std::vector<int> bid) {
  return DotProduct(product_activity_, bid);
}

std::vector<int> AuctionState::ActionToBid(Action action) const {
  return all_bids_[action];
}

void AuctionState::DoApplyActions(const std::vector<Action>& actions) {

  // Check the actions are valid.
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  for (auto p = Player{0}; p < num_players_; ++p) {
    const Action action = actions[p];
    std::vector<int> bid = ActionToBid(action);
    // TODO: If activity rule is on, you probably want to SPIEL_CHECK it here

    // TODO: If undersell rules are on, you'll need to process the queue here
    // Given bids + chance (1 random number) reorder the bids and process in order until you can't process
    // For now: no rules!

    bidseq_[p].push_back(bid);
    player_moves_++;
  }

  // Calculate aggregate demand, excess demand
  bool any_excess = false;  
  std::vector<bool> excess_demand(num_products_, false);
  std::vector<int> aggregateDemand(num_products_, 0);
  for (auto p = Player{0}; p < num_players_; ++p) {
    std::vector<int> bid = bidseq_[p].back();

    activity_[p] = DotProduct(bid, product_activity_);

    for (int j = 0; j < num_products_; ++j) {
      aggregateDemand[j] += bid[j];
      if (aggregateDemand[j] > num_licenses_[j]) {
        excess_demand[j] = true;
        any_excess = true;
      }
    }
  }
  aggregate_demands_.push_back(aggregateDemand);


  if (any_excess) {

    std::cerr << "Raising prices" << std::endl;

    // Normal case: Increment price for overdemanded items, leave other items alone
    std::vector<double> next_price = price_.back();
    for (int j = 0; j < num_products_; ++j) {
      if (excess_demand[j]) {
        next_price[j] *= (1 + increment_);
      }
    }
    price_.push_back(next_price);
  } else {
    // Demand <= supply for each item. We are finished. Let's ignore undersell for now
    for (auto p = Player{0}; p < num_players_; p++) {
      final_bids_.push_back(bidseq_[p].back());
    }
    finished_ = true;
  }
}

std::string AuctionState::ActionToString(Player player, Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  if (player != kChancePlayerId) {
    std::vector<int> bid = ActionToBid(action_id);
    return absl::StrCat("Bid for ", absl::StrJoin(bid, ","), " licenses @ ", DotProduct(bid, price_.back()), " with activity ", all_bids_activity_[action_id]);
  } else {
    if (value_.size() < num_players_) {
      return absl::StrCat("Player ", value_.size(), " was assigned values: ", absl::StrJoin(values_[value_.size()][action_id], ", "), " and a budget of ", budgets_[budget_.size()][action_id]);
    } else if (undersell_) {
      return absl::StrCat("Undersell! Allowing player ", action_id, " to drop");
    }
  }
}

Player AuctionState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

void AuctionState::DoApplyAction(Action action) {
  total_moves_++;
  
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
  // undersell_order_.push_back(action);

  // // Compute allowed number of drops
  // int required_drops = Sum(final_bids_) - num_licenses_;
  // int requested_drops = RequestedDrops()[action];
  // SPIEL_CHECK_GE(required_drops, 1);
  // SPIEL_CHECK_GE(requested_drops, 1);

  // while (required_drops > 0 && requested_drops > 0) {
  //   required_drops--;
  //   requested_drops--;
  //   final_bids_[action]--;
  // }

  // if (required_drops == 0) { // If we're done dropping
  //   finished_ = true;
  // }
}

std::vector<Action> AuctionState::LegalActions(Player player) const {
  if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
  if (player == kChancePlayerId) return LegalChanceOutcomes();
  if (player == kTerminalPlayerId) return std::vector<Action>();
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // TODO: Flag to turn activity on/off?

  // Any move weakly lower than a previous bid is allowed if it fits in budget
  std::vector<Action> actions;
  int activity_budget = activity_[player];
  
  // TODO: Are these making copies? 
  std::vector<double> price = price_.back();
  std::vector<double> value = value_[player];
  double budget = budget_[player];

  bool activity_on = true;
  bool hard_budget_on = true;
  bool positive_profit_on = false;

  for (int b = 0; b < all_bids_.size(); b++) {
    std::vector<int> bid = all_bids_[b];
    if (activity_on && activity_budget != -1 && activity_budget < all_bids_activity_[b]) {
      continue;
    }
    if (hard_budget_on && budget < DotProduct(bid, price)) {
      continue;
    }
    if (positive_profit_on && DotProduct(bid, value) < 0) {
      continue;
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
  std::string result = absl::StrCat("p", player, "|");
  if (value_.size() > player) {
    absl::StrAppend(&result, absl::StrCat("v", absl::StrJoin(value_[player], ", "), "|b", budget_[player]));  
  }
  if (!bidseq_[player].empty()) {
    for (int i = 0; i < bidseq_[player].size(); i++) {
      absl::StrAppend(&result, absl::StrCat("\n", absl::StrJoin(bidseq_[player][i], ", "), "\n"));
    }
    if (information_policy_ == kShowDemand) {
      for (int i = 0; i < aggregate_demands_.size(); i++) {
        absl::StrAppend(&result, absl::StrJoin(aggregate_demands_[i], ","));  
      }
    } else {
      for (int i = 0; i < aggregate_demands_.size(); i++) {
        std::vector<int> ad = aggregate_demands_[i];
        std::vector<std::string> hidden_demands;
        for (int j = 0; j < ad.size(); j++) {
          hidden_demands.push_back(ad[j] == num_licenses_[j] ? "=" : (ad[j] > num_licenses_[j] ? "+" : "-"));
        }
        absl::StrAppend(&result, absl::StrJoin(hidden_demands, ","));  
      }
      
    }
  }
  return result;
}

std::string DemandString(std::vector<std::vector<int>> const &demands) {
  std::string ds = "";
  for (int i = 0; i < demands.size(); i++) {
    absl::StrAppend(&ds, absl::StrJoin(demands[i], ", "), "|");
  }
  return ds;
}


std::string AuctionState::ToString() const {
  std::string result = "";
  // Player types
  for (auto p = Player{0}; p < num_players_; p++) {
      if (value_.size() > p) {
        absl::StrAppend(&result, absl::StrCat("p", p, "v", absl::StrJoin(value_[p], ","), "b", budget_[p], "\n"));  
      }
  }

  absl::StrAppend(&result, absl::StrCat("Price: ", absl::StrJoin(price_.back(), ","), "\n"));
  absl::StrAppend(&result, absl::StrCat("Round: ", price_.size(), "\n"));

  for (auto p = Player{0}; p < num_players_; p++) {
    if (!bidseq_[p].empty()) {
      absl::StrAppend(&result, absl::StrCat("Player ", p, " demanded: ", DemandString(bidseq_[p]), "\n"));
    }
  }
  if (!aggregate_demands_.empty()) {
    absl::StrAppend(&result, absl::StrCat("Aggregate demands: ", DemandString(aggregate_demands_), " "));
  }
  if (!final_bids_.empty()) {
    absl::StrAppend(&result, absl::StrCat("\nFinal bids: ", DemandString(final_bids_), " "));
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
  // int final_demand = Sum(final_bids_);
  // if (undersell_) {
  //   SPIEL_CHECK_EQ(final_demand, num_licenses_);
  // } else {
  //   SPIEL_CHECK_LE(final_demand, num_licenses_);
  // }

  std::vector<double> returns(num_players_, 0.0);
  // double price; 
  // if (undersell_ && undersell_rule_ == kUndersellPrevClock) {
  //   price = price_.end()[-2]; // If the undersell flag is triggered, let's assume the drop bids occured at start-of-round, and use the previous round's price. If not using the undersell_rule_, the more intuitive thing happens nad we just use the price
  // } else {
  //   price = price_.back();
  // }
  std::vector<double> final_price = price_.back();

  for (auto p = Player{0}; p < num_players_; p++) {
    for (int j = 0; j < num_products_; j++) {
      SPIEL_CHECK_GE(final_bids_[p][j], 0);
      // linear values
      returns[p] += (value_[p][j] - final_price[j]) * final_bids_[p][j];
    }
  }
  return returns;
}

std::unique_ptr<State> AuctionState::Clone() const {
  return std::unique_ptr<State>(new AuctionState(*this));
}

int AuctionGame::NumDistinctActions() const {
  // You can bid for [0...M_j] for any of the j products
  int product = 1;
  for (const auto& e: num_licenses_) {
    product *= e + 1;
  }
  return product;
}

std::unique_ptr<State> AuctionGame::NewInitialState() const {
  std::unique_ptr<AuctionState> state(
      new AuctionState(shared_from_this(), num_players_, num_licenses_, increment_, open_price_, product_activity_, undersell_rule_, information_policy_, allow_negative_profit_bids_, values_,  budgets_, type_probs_));
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

/**** JSON PARSING ***/

// Annoyingly hyper-sensitive to double vs int in JSON lib
double ParseDouble(json::Value val) {
  return val.IsDouble() ? val.GetDouble() : val.GetInt();
}

std::vector<double> ParseDoubleArray(json::Array a) {
  std::vector<double> d;
  for (auto t = 0; t < a.size(); t++) {
    d.push_back(ParseDouble(a[t]));
  }
  return d;
}

std::vector<int> ParseIntArray(json::Array a) {
  std::vector<int> d;
  for (auto t = 0; t < a.size(); t++) {
    d.push_back(ParseDouble(a[t]));
  }
  return d;
}

void CheckRequiredKey(json::Object obj, std::string key) {
  if (obj.find(key) == obj.end()) {
    SpielFatalError(absl::StrCat("Missing JSON key: ", key));
  }
}

AuctionGame::AuctionGame(const GameParameters& params) :
   SimMoveGame(kGameType, params) {

  std::string filename;
  if (IsParameterSpecified(game_parameters_, "filename")) {
    filename = ParameterValue<std::string>("filename");
  } else {
    std::cerr << "No file input specified. Using defaults" << std::endl;
    filename = "parameters.json";
  }

  std::cerr << "Reading from env variable CLOCK_AUCTION_CONFIG_DIR. If it is not set, there will be trouble." << std::endl;
  std::string configDir(std::getenv("CLOCK_AUCTION_CONFIG_DIR"));
  std::cerr << "CLOCK_AUCTION_CONFIG_DIR=" << configDir << std::endl;

  std::string fullPath = configDir + '/' + filename;

  std::cerr << "Parsing configuration from " << fullPath << std::endl;
  std::string string_data = file::ReadContentsFromFile(fullPath, "r");
  SPIEL_CHECK_GT(string_data.size(), 0);

  absl::optional<json::Value> v = json::FromString(string_data);
  auto object = v->GetObject();

  CheckRequiredKey(object, "players");
  auto players = object["players"].GetArray();
  num_players_ = players.size();
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);

  CheckRequiredKey(object, "opening_price");
  open_price_ = ParseDoubleArray(object["opening_price"].GetArray());

  CheckRequiredKey(object, "licenses");
  num_licenses_ = ParseIntArray(object["licenses"].GetArray());
  num_products_ = num_licenses_.size();

  CheckRequiredKey(object, "activity");
  product_activity_ = ParseIntArray(object["activity"].GetArray());

  CheckRequiredKey(object, "increment");
  increment_ = ParseDouble(object["increment"]);
  
  CheckRequiredKey(object, "undersell_rule");
  std::string undersell_rule_string = object["undersell_rule"].GetString();
  if (undersell_rule_string == "undersell_allowed") {
    undersell_rule_ = kUndersellAllowed;
  } else if (undersell_rule_string == "undersell_prev_clock") {
    undersell_rule_ = kUndersellPrevClock;
  } else if (undersell_rule_string == "undersell_standard") {
    undersell_rule_ = kUndersell;
  } else {
    SpielFatalError("Unrecognized undersell rule!");  
  }

  CheckRequiredKey(object, "information_policy");
  std::string information_policy_string = object["information_policy"].GetString();
  if (information_policy_string == "show_demand") {
    information_policy_ = kShowDemand;
  } else if (information_policy_string == "hide_demand") {
    information_policy_ = kHideDemand;
  } else {
    SpielFatalError("Unrecognized information policy rule!");  
  }

  CheckRequiredKey(object, "bidding");
  std::string bidding_string = object["bidding"].GetString();
  if (bidding_string == "unrestricted") {
    allow_negative_profit_bids_ = true;
  } else if (bidding_string == "weakly_positive_profit") {
    allow_negative_profit_bids_ = false;
  } else {
    SpielFatalError("Unrecognized bidding restrictions!");  
  }

  // Loop over players, parsing values and budgets
  std::vector<double> max_value_ = std::vector<double>(num_products_, 0.); // TODO: Fix if using
  max_budget_ = 0.;
  for (auto p = Player{0}; p < num_players_; p++) {
    auto player_object = players[p].GetObject();
    CheckRequiredKey(player_object, "type");
    auto type_array = player_object["type"].GetArray();
    std::vector<double> player_budgets;
    std::vector<std::vector<double>> player_values;
    std::vector<double> player_probs;
    for (auto t = 0; t < type_array.size(); t++) {
      auto type_object = type_array[t].GetObject();
      CheckRequiredKey(type_object, "value");
      CheckRequiredKey(type_object, "budget");
      CheckRequiredKey(type_object, "prob");
      std::vector<double> value = ParseDoubleArray(type_object["value"].GetArray());
      // for (int j = 0; j < value.size(); j++) {
      //   double v = value[j];
      //   if (v > max_value_[j]) {
      //     max_value_[j] = v;
      //   }
      // }
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
  
  std::cerr << "Done config parsing" << std::endl;
}

}  // namespace clock_auction
}  // namespace open_spiel
