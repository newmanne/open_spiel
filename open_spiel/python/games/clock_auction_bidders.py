import itertools
import numpy as np

class Bidder:

  def __init__(self, values, budget, pricing_bonus, all_bids) -> None:
    self.values = np.array(values)
    self.budget = budget
    self.pricing_bonus = pricing_bonus
    self.all_bids = all_bids
    self.bundle_values = None

  def value_for_package(package, package_index=None):
    raise NotImplementedError()
  
  def get_budget(self):
    return self.budget

  def get_pricing_bonus(self):
    return self.price_bonus

  def get_values(self):
    return self.bundle_values
  
  def get_profits(self, prices):
    return self.get_values() - prices

class LinearBidder(Bidder):

  def __init__(self, values, budget, pricing_bonus, all_bids) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids)
    self.bundle_values = np.array([[np.array(self.values) @ bid] for bid in all_bids])

  def value_for_package(self, package, package_index=None):
    return np.array(package) @ self.values

  def __str__(self) -> str:
    return f'LinearValues: {self.values} Budget: {self.budget}'

class MarginalValueBidder(Bidder):

  def __init__(self, values, budget, pricing_bonus, all_bids) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids)
    self.bundle_values = [self.value_for_package(bid) for bid in all_bids]

  def value_for_package(self, package, package_index=None):
    # TODO: Use indexing
    value = 0
    for i, quantity in enumerate(package):
      for j in range(quantity):
        value += self.values[i][j]
    return value

  def __str__(self) -> str:
    return f'MarginalValues: {self.values} Budget: {self.budget}'

class EnumeratedValueBidder(Bidder):

  def __init__(self, values, budget, pricing_bonus, all_bids) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids)
    self.bundle_values = self.values

  def value_for_package(self, package, package_index=None):
    if package_index is None:
      package_index = self.all_bids.find(package)
    return self.values[package_index]

# TODO: Duplicate of ubc_utils
def action_to_bundles(licenses):
    bids = []
    for n in licenses:
        b = []
        for i in range(n + 1):
            b.append(i)
        bids.append(b)
    actions = np.array(list(itertools.product(*bids)))
    return actions
