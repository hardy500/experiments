import re
import pandas as pd

# Part A:
# Find all the names in the following string
def names(regex=False):
  string = """Amy is 5 years old, and her sister Mary is 2 years old. Ruth and Peter, their parents, have 3 kids."""

  if regex:
    pattern = r'\b[A-Z][a-z]+\b'
    names = re.findall(pattern, string)
  else:
    string = string.replace(',', '').replace('.','').split(' ')
    names = [word for word in string if word.isalpha() and word[0].isupper()]

  assert len(names) == 4, "There are four names in the string"
  return names

# Part B:
# The dataset file in assets/grades.txt contains a line separated list of people with their grade in a class.
# Create a regex to generate a list of just those students who received a B in the course.

def grades(regex=False):
  with open('assets/grades.txt', 'r') as f:
    text = f.read()

  if regex:
    pattern = r"\b[A-Z][a-z]+ [A-Z][a-z]+: B\b"
    grade_b = re.findall(pattern, text)
  else:
    grades = text.split('\n')
    grade_b = [grade.rstrip() for grade in grades if grade.rstrip()[-1] == 'B']

  assert len(grade_b) == 16
  return grade_b

# Part C:
# Your task is to convert this into a list of dictionaries, where each dictionary looks like the following:

def logs():
  with open('assets/logdata.txt', 'r') as f:
    text = f.read()

  pattern = r'(?P<host>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) - (?P<user_name>\w*|-) \[(?P<time>.*?)\] "(?P<request>.*?)"'
  matches = re.findall(pattern, text)

  logs = []
  for match in matches:
      log = {}
      log['host'] = match[0]
      log['user_name'] = match[1]
      log['time'] = match[2]
      log['request'] = match[3]
      logs.append(log)

  assert len(logs) == 979
  return logs