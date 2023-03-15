import unittest
from city_function import get_city_country

class TestGetCityCountry(unittest.TestCase):
  def test_without_population(self):
    city_country = get_city_country('Paris', 'France')
    self.assertEqual(city_country, 'Paris, France')

  def test_with_population(self):
    city_country = get_city_country('Tokyo', 'Japan', 13929286)
    self.assertEqual(city_country, 'Tokyo, Japan - population 13929286')

if __name__ == '__main__':
    unittest.main()
