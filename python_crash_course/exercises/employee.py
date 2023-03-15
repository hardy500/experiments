#%%
class Employee:
  def __init__(self, first_name, last_name, annual_salary):
    self.first_name = first_name
    self.last_name = last_name
    self.annual_salary = annual_salary

  def give_raise(self, raise_amount=5000):
    self.annual_salary += raise_amount

if __name__ == "__main__":
  em = Employee('Bob', 'Lee', 45000)
  em.give_raise(10000)
  print(em.annual_salary)