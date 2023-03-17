#%%
from pathlib import Path
import json
import plotly.express as px

#%%

# Read data as string and convert to a python obj
path = Path('eq_data/eq_data_30_day_m1.geojson')
contents = path.read_text()
data = json.loads(contents)

#%%

# Extract all earthquakes in the dataset
eq_dicts = data['features']

#%%

mags, lons, lats, eq_titles = [], [], [], []
for s in eq_dicts:
  mag = s['properties']['mag']
  lon = s['geometry']['coordinates'][0]
  lat = s['geometry']['coordinates'][1]
  eq_title = s['properties']['title']

  lons.append(lon)
  lats.append(lat)
  mags.append(mag)
  eq_titles.append(eq_title)

#%%

# Building world map

title = "Global Earthquakes"
fig = px.scatter_geo(
  lat=lats, lon=lons, size=mags, title=title,
  color=mags,
  color_continuous_scale='Viridis',
  projection='natural earth',
  hover_name=eq_titles
)

fig.show()

