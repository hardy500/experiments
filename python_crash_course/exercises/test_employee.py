from employee import Employee
import unittest

class TestEmployee(unittest.TestCase):
  """Tests for the Employee class."""
  def setUp(self):
    """Create an employee instance for use in test methods."""
    self.employee = Employee('John', 'Doe', 60000)

  def test_give_default_raise(self):
    """Test that a default raise of $5000 works correctly."""
    self.employee.give_raise()
    self.assertEqual(self.employee.annual_salary, 65000)

  def test_give_custom_raise(self):
    """Test that a custom raise amount works correctly."""
    self.employee.give_raise(10000)
    self.assertEqual(self.employee.annual_salary, 70000)

if __name__ == '__main__':
    unittest.main()
