from flask import Flask
from kaggle_environments import make

env = make("halite", configuration={"episodeSteps": 400}, debug=True)
app = Flask(__name__)


@app.route('/')
def hello_world():
    env.run(["agent.py", "random", "random", "random"])
    return env.render(mode="html", width=800, height=600)


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=5050)
