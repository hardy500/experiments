
class Privileges:
  def  __init__(self):
    self.privileges = ['can add post', 'can delete post', 'can ban user']

  def show_privileges(self):
    print(self.privileges)