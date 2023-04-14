from flask import Flask, render_template, jsonify

app = Flask(__name__)

JOBS = [
  {
    'id':1,
    'title': 'Data Analyist',
    'location': 'Germany Hannover',
    'salary': '100,000 Euro'
  },

  {
    'id':2,
    'title': 'Data Scientist',
    'location': 'Germany Hamburg',
    'salary': '120,000 Euro'
  },

  {
    'id':4,
    'title': 'Frontend Engineer',
    'location': 'Germany Berlin',
    'salary': '130,000 Euro'
  },

  {
    'id':5,
    'title': 'Backend Engineer',
    'location': 'Germany Duisburg',
    'salary': '150,000 Euro'
  }
]

@app.route("/")
def hello_world():
  return render_template('home.html', jobs=JOBS, company_name="Toto")

@app.route("/api/jobs")
def list_jobs():
  return jsonify(JOBS)

if __name__ == "__main__":
  app.run(host='localhost', debug=True)