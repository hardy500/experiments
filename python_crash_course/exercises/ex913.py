#%%
import random

class Die:
  def __init__(self, sides=6):
    self.sides = sides + 1

  def roll_die(self):
    random_choice = random.choice([i for i in range(1, self.sides)])
    print(f"randon number: {random_choice}\ndie sides: {self.sides-1}")

if __name__ == "__main__":
  die10 = Die(sides=10)
  die20 = Die(sides=20)

  for i in range(11):
    print("roll number: ", i)
    die10.roll_die()
    die20.roll_die()
    print()
