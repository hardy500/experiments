#%%
import json

with open('answer.json', 'r') as f:
  content = json.load(f)

print(f"I know your favorite number.It’s {content}")