def get_city_country(city_name, country_name, population=''):
  if population:
    s = f"{city_name}, {country_name} - population {population}"
  else:
    s = f"{city_name}, {country_name}"
  return s