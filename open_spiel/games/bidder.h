// // Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #ifndef OPEN_SPIEL_GAMES_BIDDER_H
// #define OPEN_SPIEL_GAMES_BIDDER_H

// #include <array>
// #include <memory>
// #include <string>
// #include <vector>
// #include "open_spiel/spiel_utils.h"


// namespace open_spiel {
// namespace clock_auction {

//   class Bidder {
//     public:
//       virtual const double ValuationForPackage(std::vector<int> const &package) const = 0;
//       virtual const double GetBudget() const = 0;
//       virtual const double GetPricingBonus() const = 0;
//       virtual operator std::string() const = 0;
//   };

//   class LinearBidder : public Bidder {

//     public:
//       explicit LinearBidder(std::vector<double> values, double budget, double pricing_bonus) : values_(values), budget_(budget), pricing_bonus_(pricing_bonus) {
//       }

//       const double ValuationForPackage(std::vector<int> const &package) const override {
//         return open_spiel::DotProduct(package, values_);
//       }

//       const double GetBudget() const override {
//         return budget_;
//       }

//       const double GetPricingBonus() const override {
//         return pricing_bonus_;
//       }

//       operator std::string() const {
//           return absl::StrCat("Values:", absl::StrJoin(values_, ", "), "\n Budget: ", budget_);
//       }

//       friend std::ostream& operator<<(std::ostream& os, const LinearBidder& b);


//     private:
//       double budget_;
//       std::vector<double> values_;
//       double pricing_bonus_;

//   };

//   std::ostream& operator<<(std::ostream &strm, const LinearBidder &a) {
//     return strm << std::string(a);
//   }



// }  // namespace clock_auction
// }  // namespace open_spiel

// #endif  // OPEN_SPIEL_GAMES_BIDDER_H
