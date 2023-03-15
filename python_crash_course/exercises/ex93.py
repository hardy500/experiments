#%%
class User:
  def __init__(self, f_name, l_name, phone_number):
    self.f_name = f_name
    self.l_name = l_name
    self.phone_number = phone_number
    self.login_attempts = 0
    self.privileges = ['can add post', 'can delete post', 'can ban user']

  def describe_user(self):
    print(self.f_name, self.l_name, self.phone_number)

  def greet_user(self):
    print("Hello ", self.f_name)

  def increment_login_attempts(self):
    self.login_attempts += 1

  def reset_login_attempts(self):
    self.login_attempts = 0

if __name__ == "__main__":
  u1 = User('Bob', 'rust', 123)
