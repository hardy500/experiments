# Restaurant

# %%
class Restaurant:
  def __init__(self, name, cuisine_type):
    self.name = name
    self.cuisine_type = cuisine_type
    self.number_served = 0
    self.flavors = ['schoko', 'vanilla', 'erdbeer']

  def describe(self):
    print(f"restaurnt name: {self.name}")
    print(f"cuisine type: {self.cuisine_type}")
    self.open()

  def open(self):
    print("open")

  def set_number_served(self, number_served):
    self.number_served = number_served

  def incr_number_served(self, number_served):
    self.number_served += number_served

if __name__ == "__main__":
  res1 = Restaurant("Bob", "chicken")
  print(res1.number_served)
  res1.set_number_served(10)
  print(res1.number_served)
  res1.incr_number_served(10)
  res1.incr_number_served(10)
  print(res1.number_served)