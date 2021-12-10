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
#include "open_spiel/games/cartesian.h"

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

// Undersell rules
constexpr int kUndersellAllowed = 1;
constexpr int kUndersell = 2;

// Information policies
constexpr int kShowDemand = 1;
constexpr int kHideDemand = 2;

constexpr int kDefaultMaxRounds = 500;

int Sum(std::vector<int> v) {
  return std::accumulate(v.begin(), v.end(), 0);
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
                         /*provides_information_state_tensor=*/true,
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
  int max_rounds,
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
  std::vector<std::vector<double>> probs
) :   SimMoveState(game),
      cur_player_(kChancePlayerId), // Chance begins by setting types
      max_rounds_(max_rounds),
      finished_(false),
      tiebreaks_(tiebreaks),
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
      aggregate_demands_(),
      all_bids_activity_(),
      round_(1),
      final_bids_() {

      num_products_ = num_licenses_.size();
      // Everyone starts with lots of activity
      int max_activity = DotProduct(num_licenses, product_activity);
      activity_ = std::vector<int>(num_players_, max_activity);
      
      submitted_demand_ = std::vector<std::vector<std::vector<int>>>(num_players_, std::vector<std::vector<int>>());
      processed_demand_ = std::vector<std::vector<std::vector<int>>>(num_players_, std::vector<std::vector<int>>());
      sor_price_.push_back(open_price);
      std::vector<double> cp(num_products_, 0.);
      for (int j = 0; j < num_products_; j++) {
        cp[j] = open_price[j] *  (1 + increment_);
      }
      clock_price_.push_back(cp);
      posted_price_.push_back(open_price);

      // Enumerate bids and store a map for fast computation
      std::vector<std::vector<int>> sequences(num_products_, std::vector<int>());
      for (int j = 0; j < num_products_; j++) {
        for (int k = 0; k <= num_licenses[j]; k++) {
          sequences[j].push_back(k);
        }
      }
      product(sequences, [&vec = all_bids_](const auto &i) {
          std::vector<int> tmp;
          for (auto j : i) {
            tmp.push_back(j);
          }
          vec.push_back(tmp);
      });

      for (auto& bid: all_bids_) {
        for (int j = 0; j < num_products_; j++) {
          SPIEL_CHECK_GE(bid[j], 0);
          SPIEL_CHECK_LE(bid[j], num_licenses_[j]);
        }
        all_bids_activity_.push_back(DotProduct(bid, product_activity_));
      }

      tie_breaks_needed_ = std::vector<std::vector<Player>>(num_products_, std::vector<Player>());
      selected_order_ = std::vector<std::vector<Player>>(num_products_, std::vector<Player>());
      default_player_order_ = std::vector<std::vector<Player>>(num_products_, std::vector<Player>(num_players_, 0));
      for (int j = 0; j < num_products_; ++j) {
        for (auto p = Player{0}; p < num_players_; ++p) {
          default_player_order_[j][p] = p;
        }
      }

}

std::vector<int> AuctionState::ActionToBid(Action action) const {
  return all_bids_[action];
}


void AuctionState::ProcessBids(const std::vector<std::vector<Player>> player_order) {
  SPIEL_CHECK_EQ(player_order.size(), num_products_);
  for (int j = 0; j < num_products_; j++) {
    SPIEL_CHECK_EQ(player_order[j].size(), num_players_);
  }

  auto current_agg = aggregate_demands_.back();

  // Copy over current processed demand
  std::vector<std::vector<int>> bids;
  std::vector<std::vector<int>> requested_changes;
  // TODO: For now points is a zero vector, but possible grace period implementations would change that
  std::vector<int> points = activity_;
  for (auto p = Player{0}; p < num_players_; ++p) {
    auto last_round_holdings = processed_demand_[p].back();

    bids.push_back(last_round_holdings);
    
    std::vector<int> rq(num_products_, 0);
    for (int j = 0; j < num_products_; ++j) {
      int delta = submitted_demand_[p].back()[j] - last_round_holdings[j];
      rq[j] = delta;
      points[p] -= last_round_holdings[j] * product_activity_[j];
    }
    requested_changes.push_back(rq);
  }

  bool changed = true;

  while (changed) {
    changed = false;
    for (int j = 0; j < num_products_; ++j) {
      for (auto p : player_order[j]) {
        auto& bid = bids[p];
        auto& changes = requested_changes[p];

        // Process drops
        if (changes[j] < 0) {
          int drop_room = current_agg[j] - num_licenses_[j];
          if (drop_room > 0) {
            int amount = std::min(drop_room, -changes[j]);
            bid[j] -= amount;
            SPIEL_CHECK_GE(bid[j], 0);
            changed = true;
            points[p] += amount * product_activity_[j];
            current_agg[j] -= amount;
            changes[j] += amount;
          }
        }
        
        // Process pickups
        while (changes[j] > 0 && points[p] >= product_activity_[j]) {
          bid[j]++;
          SPIEL_CHECK_LE(bid[j], num_licenses_[j]);
          current_agg[j]++;
          changed = true;
          points[p] -= product_activity_[j];
          changes[j]--;
        }
      }
    }
  }

  // Finally, copy over submitted -> processed
  for (auto p = Player{0}; p < num_players_; ++p) {
    for (int j = 0; j < num_products_; ++j) {
      SPIEL_CHECK_GE(bids[p][j], 0);
      SPIEL_CHECK_LE(bids[p][j], num_licenses_[j]);
      SPIEL_CHECK_LE(bids[p][j], std::max(submitted_demand_[p].back()[j], processed_demand_[p].back()[j])); // Either what you asked for, or what you used to have
    }      
    processed_demand_[p].push_back(bids[p]);
    SPIEL_CHECK_EQ(processed_demand_[p].size(), submitted_demand_[p].size());
  }

  cur_player_ = kSimultaneousPlayerId;
  PostProcess();
}

void AuctionState::PostProcess() {
  // Calculate aggregate demand, excess demand
  bool any_excess = false;  
  std::vector<bool> excess_demand(num_products_, false);
  std::vector<int> aggregateDemand(num_products_, 0);
  for (auto p = Player{0}; p < num_players_; ++p) {
    auto& bid = processed_demand_[p].back();

    // Lower activity based on processed demand (TODO: May want to revisit this for grace period)
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
    // Normal case: Increment price for overdemanded items, leave other items alone
    std::vector<double> next_price(num_products_, 0);
    std::vector<double> next_clock(num_products_, 0);
    for (int j = 0; j < num_products_; ++j) {
      if (excess_demand[j]) {
        next_price[j] = clock_price_.back()[j];
      } else {
        next_price[j] = sor_price_.back()[j];
      }
      next_clock[j] = next_price[j] * (1 + increment_);
    }
    posted_price_.push_back(next_price);
    sor_price_.push_back(next_price);
    clock_price_.push_back(next_clock);
    round_++;
  } else {
    // Demand <= supply for each item. We are finished.
    finished_ = true;
  }
}

bool AuctionState::DetermineTiebreaks() {
  /* Step 1: Figure out for each product whether we may be in a tie-breaking situation. Note that we can have false positives, they "just" make the game bigger. 
    *         One necessary condition: At least two people want to drop the same product AND the combined dropping will take the product below supply
              Note that this doesn't consider demand that might be added to the product - this could resolve the issue, but if that pick-up can only be processed conditional on another drop, it gets more complicated... We ignore this for now.
    */
  tie_breaks_needed_.clear();
  selected_order_.clear();
  tie_break_index_ = 0;
  bool tie_breaking_not_needed = true;
  std::vector<std::vector<int>> drops_per_product; // First index is product, second is player. Positive numbers. 0 if not a drop
  auto current_agg = aggregate_demands_.back();

  for (int j = 0; j < num_products_; ++j) {
    std::vector<int> drops;
    for (auto p = Player{0}; p < num_players_; ++p) {
      int delta = submitted_demand_[p].back()[j] - processed_demand_[p].back()[j];   
      drops.push_back(delta < 0 ? -delta : 0);
    }
    drops_per_product.push_back(drops);
  }

  for (int j = 0; j < num_products_; ++j) {
    std::vector<Player> tiebreaks;
    if (current_agg[j] - Sum(drops_per_product[j]) < num_licenses_[j]) {
      for (auto p = Player{0}; p < num_players_; ++p) {
        if (drops_per_product[j][p] > 0) {
          tiebreaks.push_back(p);
        }
      }
    }
    if (tiebreaks.size() <= 1) {
      tiebreaks.clear(); // No tie-breaking if only one person dropping
    } else {
      if (tie_breaking_not_needed) { // First time we've realized we'll need to tie-break. Set the intial index appropriately
        tie_breaking_not_needed = false; 
        tie_break_index_ = j;
      }
    }
    tie_breaks_needed_.push_back(tiebreaks);
  }
  return tie_breaking_not_needed;
}

void AuctionState::DoApplyActions(const std::vector<Action>& actions) {
  // Check the actions are valid.
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  for (auto p = Player{0}; p < num_players_; ++p) {
    SPIEL_CHECK_EQ(processed_demand_[p].size(), submitted_demand_[p].size());
    SPIEL_CHECK_EQ(round_ - 1, submitted_demand_[p].size());
    const Action action = actions[p];
    auto bid = ActionToBid(action);
    SPIEL_CHECK_GE(activity_[p], all_bids_activity_[action]);
    submitted_demand_[p].push_back(bid);
  }

  // Demand processing
  if (round_ == 1 || undersell_rule_ == kUndersellAllowed) {
    // Just copy it straight over
    for (auto p = Player{0}; p < num_players_; ++p) {
      auto bid = submitted_demand_[p].back();
      processed_demand_[p].push_back(bid);
    }
    PostProcess();
  } else if (undersell_rule_ == kUndersell) {
    bool tiebreaks_not_needed = true;
    if (tiebreaks_ && round_ > 1) {
      tiebreaks_not_needed = DetermineTiebreaks();
    }
    if (tiebreaks_not_needed) { // No chance node required. Just get on with the game
      ProcessBids(default_player_order_); 
    } else {
      cur_player_ = kChancePlayerId;
    }
  } else {
      SpielFatalError("Unknown undersell");  
  }
}

std::string AuctionState::ActionToString(Player player, Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  if (player != kChancePlayerId) {
    std::vector<int> bid = ActionToBid(action_id);
    return absl::StrCat("Bid for ", absl::StrJoin(bid, ","), " licenses @ $", DotProduct(bid, posted_price_.back()), " with activity ", all_bids_activity_[action_id]);
  } else {
    if (value_.size() < num_players_) {
      return absl::StrCat("Player ", value_.size(), " was assigned values: ", absl::StrJoin(values_[value_.size()][action_id], ", "), " and a budget of ", budgets_[budget_.size()][action_id]);
    } else {
      return "Tie-break";
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
  SPIEL_CHECK_TRUE(IsChanceNode());

  if (round_ == 1) {
    if (value_.size() < num_players_) { // Chance node assigns a value and budget to a player
      auto player_index = value_.size();
      SPIEL_CHECK_GE(player_index, 0);
      value_.push_back(values_[player_index][action]); 
      budget_.push_back(budgets_[player_index][action]);
    } 

    if (value_.size() == num_players_) { // All of the assignments have been made
      cur_player_ = kSimultaneousPlayerId;
    }
  } else {
    // Pad if needed
    while (selected_order_.size() < tie_break_index_) {
      std::vector<Player> order;
      for (auto p = Player{0}; p < num_players_; ++p) {
        order.push_back(p);    
      }
      selected_order_.push_back(order);
    }


    std::vector<Player> order = tie_breaks_needed_[tie_break_index_];
    // Assign ordering for this index
    for (int i = 0; i < action; i++) {
      std::next_permutation(order.begin(), order.end());
    }

    // Pad with players that aren't in the list
    for (auto p = Player{0}; p < num_players_; ++p) {
      if (std::find(order.begin(), order.end(), p) == order.end()) { // Not in vector
        order.push_back(p);    
      }
    }
    selected_order_.push_back(order);
 
    tie_break_index_++;
    while (tie_break_index_ < num_products_) {
      if (tie_breaks_needed_[tie_break_index_].size() == 0) {
        std::vector<Player> order = default_player_order_[tie_break_index_];
        selected_order_.push_back(order);
        tie_break_index_++;
      } else {
        break; // We need to actually do this one and it will require another chance node
      }
    }
    if (tie_break_index_ == num_products_) { // Done with chance nodes
      ProcessBids(selected_order_);
    }
  }
}

bool IsZeroBid(std::vector<int> const &bid) {
  return std::all_of(bid.begin(), bid.end(), [](int i) { return i==0; });
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

  if (submitted_demand_[player].size() > 0 && IsZeroBid(submitted_demand_[player].back())) {
    // Don't need to recalculate
    actions.push_back(0);
    return actions;
  }

  int activity_budget = activity_[player];

  auto& price = posted_price_.back();
  auto& value = value_[player];
  double budget = budget_[player];

  bool activity_on = true;
  bool hard_budget_on = true;
  bool positive_profit_on = false;

  bool all_bad = true;
  for (int b = 0; b < all_bids_.size(); b++) {
    auto& bid = all_bids_[b];

    /* Note we a) use posted prices and b) assume all drops go through. A more sophisticated bidder might think differently (e.g., try to fulfill budget in expectation)
     * Consider e.g. if you drop a product you might get stuck! So you can wind up over your budget if your drop fails
     * Also consider that if you drop a product and get stuck, you only pay SoR on that product
     */

    double bid_price = DotProduct(bid, price); 
    double profit = DotProduct(bid, value) - bid_price;
    // TODO: Here is another place where linear valuations creep in, might be better to abstract into a class
    bool non_negative_profit = profit >= 0;
    bool positive_profit = profit > 0;

    if (activity_on && activity_budget != -1 && activity_budget < all_bids_activity_[b]) {
      continue;
    }
    if (hard_budget_on && budget < bid_price) {
      continue;
    }
    if (positive_profit_on && !non_negative_profit) {
      continue;
    }

    if (positive_profit) {
      // There is some legal bid you can make that would benefit you
      all_bad = false;
    }

    actions.push_back(b);
  }

  if (all_bad) {
    // If you have no way to make a profit ever going forwards, just drop out. Helps minimize game size
    actions.clear();
    actions.push_back(0);
  }
  return actions;
}

std::vector<std::pair<Action, double>> AuctionState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  ActionsAndProbs valuesAndProbs;

  std::vector<double> probs;
  if (value_.size() < num_players_) { // Chance node is assigning a type
    auto player_index = value_.size();
    probs = type_probs_[player_index];
  } else { // Chance node is breaking a tie
    int chance_outcomes_required = Factorial(tie_breaks_needed_[tie_break_index_].size());
    for (int i = 0; i < chance_outcomes_required; i++) {
      probs.push_back(1. / chance_outcomes_required);
    }
  }

  for(int i = 0; i < probs.size(); i++) {
    valuesAndProbs.push_back({i, probs[i]});
  }
  return valuesAndProbs;
}

std::vector<std::string> AuctionState::ToHidden(const std::vector<int>& demand) const {
  std::vector<std::string> hidden_demands;
  for (int j = 0; j < demand.size(); j++) {
    hidden_demands.push_back(demand[j] == num_licenses_[j] ? "=" : (demand[j] > num_licenses_[j] ? "+" : "-"));
  }
  return hidden_demands;
}

std::string AuctionState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string result = absl::StrCat("Player ", player, "\n");
  if (value_.size() > player) {
    absl::StrAppend(&result, absl::StrCat("Values:", absl::StrJoin(value_[player], ", "), "\n Budget: ", budget_[player]), "\n");  
  }
  if (!submitted_demand_[player].empty()) {
    for (int i = 0; i < submitted_demand_[player].size(); i++) {
      absl::StrAppend(&result, absl::StrCat(absl::StrJoin(submitted_demand_[player][i], ", "), i == submitted_demand_[player].size() - 1 ? "" : " | "));
    }

    if (undersell_rule_ == kUndersell) {
      absl::StrAppend(&result, "\n");
      for (int i = 0; i < processed_demand_[player].size(); i++) {
        absl::StrAppend(&result, absl::StrCat(absl::StrJoin(processed_demand_[player][i], ", "), i == processed_demand_[player].size() - 1 ? "" : " | "));
      }
    }

    absl::StrAppend(&result, "\n");
    if (information_policy_ == kShowDemand) {
      for (int i = 0; i < aggregate_demands_.size(); i++) {
        absl::StrAppend(&result, absl::StrCat(absl::StrJoin(aggregate_demands_[i], ", "), i == aggregate_demands_.size() - 1 ? "" : " | "));  
      }
    } else if (information_policy_ == kHideDemand) {
      for (int i = 0; i < aggregate_demands_.size(); i++) {
        absl::StrAppend(&result, absl::StrCat(absl::StrJoin(ToHidden(aggregate_demands_[i]), ", "), i == aggregate_demands_.size() - 1 ? "" : " | "));  
      }
    } else {
      SpielFatalError("Unknown info policy");
    }
  }
  return result;
}

std::string DemandString(std::vector<std::vector<int>> const &demands) {
  std::string ds = "";
  for (int i = 0; i < demands.size(); i++) {
    absl::StrAppend(&ds, absl::StrJoin(demands[i], ", "), i == demands.size() - 1 ? "" : " | ");
  }
  return ds;
}


std::string AuctionState::ToString() const {
  std::string result = "";
  // Player types
  for (auto p = Player{0}; p < num_players_; p++) {
      if (value_.size() > p) {
        absl::StrAppend(&result, absl::StrCat("Player ", p, "\n", "Values: ", absl::StrJoin(value_[p], ", "), "\n", "Budget: ", budget_[p], "\n"));  
      }
  }

  absl::StrAppend(&result, absl::StrCat("Price: ", absl::StrJoin(posted_price_.back(), ", "), "\n"));
  absl::StrAppend(&result, absl::StrCat("Round: ", round_, "\n"));

  for (auto p = Player{0}; p < num_players_; p++) {
    if (!processed_demand_[p].empty()) {
      absl::StrAppend(&result, absl::StrCat("Player ", p, " processed: ", absl::StrJoin(processed_demand_[p].back(), ", "), "\n"));
    }
  }
  if (!aggregate_demands_.empty()) {
    absl::StrAppend(&result, absl::StrCat("Aggregate demands: ", absl::StrJoin(aggregate_demands_.back(), ", ")), "\n");
  }
  if (finished_) {
    std::vector<std::vector<int>> fb;
    for (auto p = Player{0}; p < num_players_; p++) {
      fb.push_back(processed_demand_[p].back());
    }
    absl::StrAppend(&result, absl::StrCat("Final bids: ", DemandString(fb)));
  }

  return result;
}

bool AuctionState::IsTerminal() const { 
  if (round_ >= kDefaultMaxRounds) {
    std::cerr << "Number of rounds exceeded limit of " << kDefaultMaxRounds << "! Terminating prematurely...\n" << std::endl;
  }
  return finished_ || round_ >= kDefaultMaxRounds; 
}

std::vector<double> AuctionState::Returns() const {
  std::vector<double> returns(num_players_, finished_ ? 0.0 : -9999999); // Return large negative number to everyone if the game went on forever
  if (finished_) {
    auto& final_price = posted_price_.back();

    for (auto p = Player{0}; p < num_players_; p++) {
      auto& final_bid = processed_demand_[p].back();
      for (int j = 0; j < num_products_; j++) {
        SPIEL_CHECK_GE(final_bid[j], 0);
        // linear values
        returns[p] += (value_[p][j] - final_price[j]) * final_bid[j];
     }
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
      new AuctionState(shared_from_this(), num_players_, max_rounds_, num_licenses_, increment_, open_price_, product_activity_, undersell_rule_, information_policy_, allow_negative_profit_bids_, tiebreaks_, values_,  budgets_, type_probs_));
  return state;
}

int AuctionGame::MaxChanceOutcomes() const { 
  return max_chance_outcomes_;
}

int AuctionGame::MaxGameLength() const {
  return kDefaultMaxRounds * num_players_; // Probably much shorter...
}

std::vector<int> AuctionGame::ObservationTensorShape() const {
  SpielFatalError("Unimplemented ObservationTensorShape");  
}

void AuctionState::ObservationTensor(Player player, absl::Span<float> values) const {
  SpielFatalError("Unimplemented ObservationTensor");
}


int AuctionGame::SizeHelper(int rounds) const {
    int size_required = 
      num_players_ + // player encoding
      1 + // budget
      num_products_ + // values
      num_products_ * rounds + // submitted demand
      num_products_ * rounds + // proceseed demand
      num_products_ * rounds + // aggregate demand
      num_products_ * rounds;  // posted price;
      return size_required;
}


std::vector<int> AuctionGame::InformationStateTensorShape() const {
  return {SizeHelper(max_rounds_)};
}

void AuctionState::InformationStateTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_LE(round_, max_rounds_);

  int highest_round = max_rounds_;
  int offset = 0;
  std::fill(values.begin(), values.end(), 0.);

  // 1-hot player encoding
  values[player] = 1;
  offset += num_players_;

  // Budget encoding - player's budget
  if (budget_.size() > player) {
    values[offset] = budget_[player];
  }
  offset += 1;

  // Values encoding - player's value for each item
  if (value_.size() > player) {
    for (int i = 0; i < value_[player].size(); i++) {
      values[offset + i] = value_[player][i];
    }
  }
  offset += num_products_;

  // Submitted demand encoding - demand submitted in each round
  if (!submitted_demand_[player].empty()) {
    for (int i = 0; i < submitted_demand_[player].size(); i++) {
      for (int j = 0; j < num_products_; j++) {
        values[offset + i * num_products_ + j] = submitted_demand_[player][i][j];
      } 
    }
  }
  offset += highest_round * num_products_;

  // Processed demand encoding (could turn this off w/o undersell)
  if (!processed_demand_[player].empty()) {
    for (int i = 0; i < processed_demand_[player].size(); i++) {
      for (int j = 0; j < num_products_; j++) {
        values[offset + i * num_products_ + j] = processed_demand_[player][i][j];
      } 
    }
  }
  offset += highest_round * num_products_;

  // History encoding - what you observed after each round
  for (int i = 0; i < aggregate_demands_.size(); i++) {
    for (int j = 0; j < num_products_; j++) {
        int val = aggregate_demands_[i][j];
        if (information_policy_ == kHideDemand) {
          val = val > num_licenses_[j] ? 1 : val == num_licenses_[j] ? 0 : -1;
        }
        values[offset + i * num_products_ + j] = val;
    }
  }
  offset += highest_round * num_products_;

  // Price encoding - (this is derivable, but let's give it to the NN). Just posted for now
  // std::vector<std::vector<double>> posted_price_;
  for (int i = 0; i < posted_price_.size(); i++) {
    for (int j = 0; j < num_products_; j++) {
      values[offset + i * num_products_ + j] = posted_price_[i][j];
    } 
  }

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

bool ContainsKey(json::Object obj, std::string key) {
  return obj.find(key) != obj.end();
}

void CheckRequiredKey(json::Object obj, std::string key) {
  if (!ContainsKey(obj, key)) {
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

  std::string fullPath;
  if (!filename.rfind("/", 0) == 0) {
    std::cerr << "Reading from env variable CLOCK_AUCTION_CONFIG_DIR. If it is not set, there will be trouble." << std::endl;
    std::string configDir(std::getenv("CLOCK_AUCTION_CONFIG_DIR"));
    std::cerr << "CLOCK_AUCTION_CONFIG_DIR=" << configDir << std::endl;
    fullPath = configDir + '/' + filename;
  } else {
    fullPath = filename;
  }

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

  if (open_price_.size() != num_products_) {
    SpielFatalError("Mistmatched size of open price and number of licences!");  
  }

  CheckRequiredKey(object, "activity");
  product_activity_ = ParseIntArray(object["activity"].GetArray());

  CheckRequiredKey(object, "increment");
  increment_ = ParseDouble(object["increment"]);

  // Actually hard to calculate, so let's just do this (clearly wrong) thing for now and ask you to supply it. If you aren't using deep CFR, this does nothing
  if (ContainsKey(object, "max_rounds")) {
    max_rounds_ = ParseDouble(object["max_rounds"]);
  } else {
    max_rounds_ = kDefaultMaxRounds; // Doesn't really matter for anything
  }

  if (ContainsKey(object, "tiebreaks")) {
    tiebreaks_ = object["tiebreaks"].GetBool();
  } else {
    tiebreaks_ = true;
  }

  CheckRequiredKey(object, "undersell_rule");
  std::string undersell_rule_string = object["undersell_rule"].GetString();
  if (undersell_rule_string == "undersell_allowed") {
    undersell_rule_ = kUndersellAllowed;
  } else if (undersell_rule_string == "undersell") {
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
      if (value.size() != num_products_) {
        SpielFatalError("Mistmatched size of value and number of licences!");  
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
  lengths.push_back(Factorial(num_players_) * num_products_);
  max_chance_outcomes_ = *std::max_element(lengths.begin(), lengths.end());
  

  std::cerr << "Done config parsing" << std::endl;
}

}  // namespace clock_auction
}  // namespace open_spiel
