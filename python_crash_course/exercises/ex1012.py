#%%
import json

try:
  with open('answer.json', 'r') as f:
    content = json.load(f)
    print(f"I know your favorite number.It’s {content}")
except FileNotFoundError:
  answer = input("What is your favorite number? ")
  with open("answer.json", "w") as f:
    json.dump(answer, f)