#%%
import json

answer = input("What is your favorite number? ")
with open("answer.json", "w") as f:
  json.dump(answer, f)