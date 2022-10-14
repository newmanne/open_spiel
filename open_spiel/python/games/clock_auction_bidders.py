import numpy as np

class Bidder:

  def __init__(self, values, budget, pricing_bonus, all_bids, drop_out_heuristic) -> None:
    self.values = np.array(values)
    self.budget = budget
    self.pricing_bonus = pricing_bonus
    self.all_bids = all_bids
    self.bundle_values = None
    self.drop_out_heuristic = drop_out_heuristic

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

  def __init__(self, values, budget, pricing_bonus, all_bids, drop_out_heuristic) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids, drop_out_heuristic)
    self.bundle_values = all_bids @ self.values

  def value_for_package(self, package, package_index=None):
    return np.array(package) @ self.values

  def __str__(self) -> str:
    return f'LinearValues: {self.values} Budget: {self.budget}'

class MarginalValueBidder(Bidder):

  def __init__(self, values, budget, pricing_bonus, all_bids, drop_out_heuristic) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids, drop_out_heuristic)
    self.bundle_values = [self.value_for_package(bid) for bid in all_bids]

  def value_for_package(self, package, package_index=None):
    value = 0
    for i, quantity in enumerate(package):
      value += self.values[i][:quantity].sum()
    return value

  def __str__(self) -> str:
    return f'MarginalValues: {self.values} Budget: {self.budget}'

class EnumeratedValueBidder(Bidder):

  def __init__(self, values, budget, pricing_bonus, all_bids, drop_out_heuristic) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids, drop_out_heuristic)
    self.bundle_values = self.values

  def value_for_package(self, package, package_index=None):
    if package_index is None:
      package_index = np.where((self.all_bids == package).all(axis=1))[0][0]
    return self.values[package_index]

  def __str__(self) -> str:
    return f'EnumeratedValues: {self.values} Budget: {self.budget}'
