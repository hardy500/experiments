#%%
from ex93 import User
from ex98 import Privileges

class Admin(User, Privileges):
  def __init__(self):
    super().__init__(f_name='Bob', l_name='Lee', phone_number='1234')
    self.privileges = self.show_privileges()

if __name__ == "__main__":
  admin = Admin()
  admin.privileges