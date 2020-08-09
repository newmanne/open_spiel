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

#include "open_spiel/games/auction.h"

#include <algorithm>
#include <array>
#include <utility>

#include "open_spiel/game_parameters.h"

namespace open_spiel {
namespace auction {

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
constexpr double v1 = 125;
constexpr double v2 = 150;
// Budgets. You cannot bid above your budget.
constexpr double b1 = 350;
constexpr double b2 = 400;

constexpr double move_limit = 4;

// Facts about the game
const GameType kGameType{/*short_name=*/"auction",
                         /*long_name=*/"Auction",
                         GameType::Dynamics::kSequential,
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

AuctionState::AuctionState(std::shared_ptr<const Game> game, int num_licenses, 
double increment, double open_price)
    : State(game),
      cur_player_(kChancePlayerId),
      total_moves_(0),
      player_moves_(0),
      underdemand_(false),
      num_licenses_(num_licenses),
      increment_(increment),
      bidseq_(),
      bidseq_str_() {
      for (auto p = Player{0}; p < num_players_; p++) {
        std::vector<int> demand;
        bidseq_.push_back(demand);
      }
      price_.push_back(open_price);
  }


std::string AuctionState::ActionToString(Player player,
                                           Action action_id) const {
  if (player != kChancePlayerId) {
    return absl::StrCat("Player ", player, " bid for ", action_id, " licenses");
  } else {
    if (value_.size() < num_players_) {
      return absl::StrCat("Player ", value_.size(), " value: ", action_id);
    } else {
      return absl::StrCat("Player ", budget_.size(), " budget: ", action_id);
    }
    
    // return absl::StrCat("Player ", budget_.size(), " budget: ", action_id);
    // Assignment chance nodes of type budget/value
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
  if (IsChanceNode()) {
      if (value_.size() < num_players_) {
        value_.push_back(action ? v1 : v2); 
        budget_.push_back(action ? b1 : b2);
      } 
      if (budget_.size() == num_players_) {
        cur_player_ = 0;
      }
  } else {
    // Check for legal actions.
    if (!bidseq_[cur_player_].empty() && action > bidseq_[cur_player_].back()) {
      SpielFatalError(absl::StrCat("Illegal action. ", action,
                                   " should be weakly less than ",
                                   bidseq_[cur_player_].back()));
    }
    bidseq_[cur_player_].push_back(action);
    if (cur_player_ == num_players_ - 1) { // last to act
      int s = 0;
      for (auto p = Player{0}; p < num_players_; p++) {
        s += bidseq_[p].back();
      }
      if (s <= num_licenses_) {
        // TODO: Return to kChancePlayer if s is strictly less and only let one player drop
        underdemand_ = true;
      } else {
        // Increment price
        price_.push_back(price_.back() * (1 + increment_));
      }
    }
    cur_player_ = NextPlayerRoundRobin(cur_player_, num_players_);
    player_moves_++;
  }
  total_moves_++;
}

std::vector<Action> AuctionState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    std::vector<Action> values(2); // two possibile types
    std::iota(values.begin(), values.end(), 0);
    return values;
  }

  std::vector<Action> actions;

  // Any move weakly lower than a previous bid is allowed if it fits in budget
  int limit = num_licenses_;
  if (bidseq_[cur_player_].size() > 0) {
    limit = bidseq_[cur_player_].back();
  }
  for (int b = 0; b <= limit; b++) {
    if (price_.back() * b > budget_[cur_player_]) {
      break;
    }
    actions.push_back(b);
  }

  return actions;
}

std::vector<std::pair<Action, double>> AuctionState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());

  ActionsAndProbs valuesAndProbs;
  if (value_.size() < num_players_) {
    // TODO: Make this neater
    valuesAndProbs.push_back(std::make_pair(0, 1. / 2));
    valuesAndProbs.push_back(std::make_pair(1, 1. / 2));
  } else if (budget_.size() < num_players_) {
    valuesAndProbs.push_back(std::make_pair(0, 1. / 2));
    valuesAndProbs.push_back(std::make_pair(1, 1. / 2));
  } 
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
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, "\n");
    for (int i = 0; i < bidseq_[p].size(); i++) {
      absl::StrAppend(&result, absl::StrCat("r", i + 1, " p", p, " bid ", bidseq_[p][i], " "));
    }
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

bool AuctionState::IsTerminal() const { return underdemand_ || player_moves_ >= move_limit; }

std::vector<double> AuctionState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);
  if (!underdemand_) {
    return returns; // Everyone gets 0, this was caused by move limit
  }
  for (auto p = Player{0}; p < num_players_; p++) {
    returns[p] = (value_[p] - price_.back()) * bidseq_[p].back();
  }

  return returns;
}

void AuctionState::InformationStateTensor(Player player,
                                            std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

}

void AuctionState::ObservationTensor(
    Player player, std:: vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

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
  // budget_types_ = ParameterValue<std::vector<double>>("budget_types");
  // value_types_ = ParameterValue<std::vector<double>>("value_types");
}

int AuctionGame::NumDistinctActions() const {
  return num_licenses_ + 1; // Bid for any number of licenes from 0 to M
}

std::unique_ptr<State> AuctionGame::NewInitialState() const {
  std::unique_ptr<AuctionState> state(
      new AuctionState(shared_from_this(), num_licenses_, increment_, open_price_));
  return state;
}

int AuctionGame::MaxChanceOutcomes() const { return 2; }

int AuctionGame::MaxGameLength() const {
  return 2000; // TODO: This is just a large number I put in. I assume it isn't being used by the algorithms we are interested in using.
}

std::vector<int> AuctionGame::InformationStateTensorShape() const {
  // One-hot encoding for the player number.
  // One-hot encoding for each die (max_dice_per_player_ * sides).
  // One slot(bit) for each legal bid.
  // One slot(bit) for calling liar. (Necessary because observations and
  // information states need to be defined at terminals)
  return {num_players_};
}

std::vector<int> AuctionGame::ObservationTensorShape() const {
  // One-hot encoding for the player number.
  // One-hot encoding for each die (max_dice_per_player_ * sides).
  // One slot(bit) for the num_players_ last legal bid.
  // One slot(bit) for calling liar. (Necessary because observations and
  // information states need to be defined at terminals)
  return {num_players_};
}

}  // namespace auction
}  // namespace open_spiel
