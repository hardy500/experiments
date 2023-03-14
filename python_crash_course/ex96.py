#%%
from ex91 import Restaurant

class IceCreamStand(Restaurant):
  def __init__(self):
    super().__init__("Bob", "Ice Cream")

  def display_flavors(self):
    print(self.flavors)


ice = IceCreamStand()
ice.display_flavors()