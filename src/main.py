import logging
import os

from flask import Flask
from flask import request

import logic

import numpy as np


app = Flask(__name__)


class NNUE:
    "An efficiently updatable neural network for evaluation"

    def __init__(self, ft_weight, ft_bias, l1_weight, l1_bias, l2_weight, l2_bias):
        self.ft_weight = ft_weight
        self.ft_bias = ft_bias
        self.l1_weight = l1_weight
        self.l1_bias = l1_bias
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias

        self.accumulator = None
        self.refresh_accumulator([])

    def foward(self, features=None):
        accumulator = self._ft(features) if features else self.accumulator
        l1_x = self._relu(accumulator)
        l2_x = self._relu(self._l1(l1_x))

        return self._l2(l2_x)

    def refresh_accumulator(self, active_features):
        self.accumulator = self.ft_bias.copy()
        for active_feature in active_features:
            self.accumulator += self.ft_weight[:, active_feature]

    def update_accumulator(self, removed_features, added_features):
        for removed_feature in removed_features:
            self.accumulator -= self.ft_weight[:, removed_feature]
        for added_feature in added_features:
            self.accumulator += self.ft_weight[:, added_feature]

    def _ft(self, features):
        return self.ft_weight @ features + self.ft_bias

    def _l1(self, features):
        return self.l1_weight @ features + self.l1_bias

    def _l2(self, features):
        return self.l2_weight @ features + self.l2_bias

    def _relu(self, features):
        return np.maximum(0, features)


@app.get("/")
def handle_info():
    """
    This function is called when you register your Battlesnake on play.battlesnake.com
    See https://docs.battlesnake.com/guides/getting-started#step-4-register-your-battlesnake
    """
    print("INFO")
    return logic.get_info()


@app.post("/start")
def handle_start():
    """
    This function is called everytime your Battlesnake enters a game.
    It's purely for informational purposes, you don't have to make any decisions here.
    request.json contains information about the game that's about to be played.
    """
    data = request.get_json()
    
    logic.choose_start(data)

    print(f"{data['game']['id']} START")
    return "ok"


@app.post("/move")
def handle_move():
    """
    This function is called on every turn and is how your Battlesnake decides where to move.
    Valid moves are "up", "down", "left", or "right".
    """
    data = request.get_json()

    # TODO - look at the logic.py file to see how we decide what move to return!
    move = logic.choose_move(data)
    shout = logic.choose_shout(data, move)

    return {"move": move, "shout": shout}


@app.post("/end")
def handle_end():
    """
    This function is called when a game your Battlesnake was in has ended.
    It's purely for informational purposes, you don't have to make any decisions here.
    """
    data = request.get_json()
    
    logic.choose_end(data)

    print(f"{data['game']['id']} END")
    return "ok"


@app.after_request
def identify_server(response):
    response.headers["Server"] = "BattlesnakeOfficial/starter-snake-python"
    return response


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8080"))

    print(f"\nRunning Battlesnake server at http://{host}:{port}")
    app.env = 'development'
    app.run(host=host, port=port, debug=True)
