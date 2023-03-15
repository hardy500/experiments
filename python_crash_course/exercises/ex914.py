#%%
import random
import string

pool = [random.randint(0, 9) for _ in range(10)] \
  + [random.choice(string.ascii_letters) for _ in range(5)]
samples = random.sample(pool, k=4)
print(f"Any ticket matching this {samples} win")
