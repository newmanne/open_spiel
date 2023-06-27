import numpy as np

def quasilinear_utility(profit):
  return profit

def risk_averse_utility(profit, alpha=1):
  if alpha == 0:
    return profit
  else:
    return (1 - np.exp(-alpha * profit)) / alpha

UTILITY_FUNCTIONS = {
  'quasilinear': quasilinear_utility,
  'risk_averse': risk_averse_utility
}

class Bidder:

  def __init__(self, values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config) -> None:
    self.values = np.array(values)
    self.budget = budget
    self.pricing_bonus = pricing_bonus
    self.all_bids = all_bids
    self.bundle_values = None
    self.drop_out_heuristic = drop_out_heuristic
    self.straightforward = False

    self.utility_function = UTILITY_FUNCTIONS[utility_function_config.pop('name')]
    self.utility_function_kwargs = utility_function_config
    

  def value_for_package(package, package_index=None):
    raise NotImplementedError()
  
  def get_budget(self):
    return self.budget

  def get_pricing_bonus(self):
    return self.price_bonus

  def get_values(self):
    return self.bundle_values
  
  def get_profits(self, prices):
    return self.get_values() - (self.all_bids @ np.asarray(prices))

  def get_utility(self, profit):
    return self.utility_function(profit, **self.utility_function_kwargs)
  
class LinearBidder(Bidder):

  def __init__(self, values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config)
    self.bundle_values = all_bids @ self.values

  def value_for_package(self, package, package_index=None):
    return np.array(package) @ self.values

  def __str__(self) -> str:
    return f'LinearValues: {self.values} Budget: {self.budget}'

class MarginalValueBidder(Bidder):

  def __init__(self, values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config)
    self.bundle_values = [self.value_for_package(bid) for bid in all_bids]

  def value_for_package(self, package, package_index=None):
    value = 0
    for i, quantity in enumerate(package):
      value += self.values[i][:quantity].sum()
    return value

  def __str__(self) -> str:
    return f'MarginalValues: {self.values} Budget: {self.budget}'

class EnumeratedValueBidder(Bidder):

  def __init__(self, values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config, name, straightforward=False) -> None:
    super().__init__(values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config)
    self.bundle_values = self.values
    self.name = name
    self.straightforward = straightforward

  def value_for_package(self, package, package_index=None):
    if package_index is None:
      package_index = np.where((self.all_bids == package).all(axis=1))[0][0]
    return self.values[package_index]

  def __str__(self) -> str:
    if self.name is not None:
      return self.name
    return f'EnumeratedValues: {self.values} Budget: {self.budget}'
