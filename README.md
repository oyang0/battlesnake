# Nuppeppou

This is Nuppeppou, a Battlesnake implemented in Python. It's deployed with [Replit](https://repl.it).

## Technologies Used

* [Python3](https://www.python.org/)
* [Flask](https://flask.palletsprojects.com/)


## Profile

* [Daedalean](https://play.battlesnake.com/u/daedalean/)

---

## Customizations

Nuppeppou presently uses these settings for personalizing its appearance:

```python
return {
    "apiversion": "1",
    "author": "daedalean",
    "color": "#E80978",
    "head": "pixel",
    "tail": "pixel",
    "version": "0.0.1-beta",
}

```

## Behavior

On the start of each game, Nuppeppou initializes or recycles an [efficiently updatable neural network](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network), except when 32 efficiently updatable neural networks were initialized and none are available for recycling. 

The input feature set for the efficiently updatable neural networks are (game_rules, board_size, x, y, piece_type) tuples. There are 6 game rules: Standard, Constrictor, Battlesnake Royale, Wrapped, Solo, and Squad. There are 3 board sizes: small (7×7), medium (11×11), and large (19×19). There are 20 piece types: your head, your body piece moving left, your body piece moving right, your body piece moving down, your body piece moving up, your body piece under a piece, a squadmate's head, a squadmate's body piece moving left, a squadmate's body piece moving right, a squadmate's body piece moving down, a squadmate's body piece moving up, a squadmate's body piece under a piece, an opponent's head, an opponent's body piece moving left, an opponent's body piece moving right, an opponent's body piece moving down, an opponent's body piece moving up, an opponent's body piece under a piece, food, and hazard. Therefore, there are 6×7×7×20 such tuples for the small board size, 6×11×11×20 such tuples for the medium board size, and 6×19×19×20 such tuples for the large board size, for a sum of 6×7×7×20+6×11×11×20+6×19×19×20=63720 such tuples. If the game rules are G and the board size is B and there is a piece type P on the square with coordinates (x, y), then we set the input (G, B, x, y, P) to 1. Otherwise, we set it to 0. 

The layers used in the efficiently updatable neural networks are three linear layers, 63720→512, 512→512, 512→4. All layers are linear and all hidden neurons use the ReLU activation function.

The efficiently updatable neural networks were trained with supervised learning methods. In particular, a training data set, validation data set, test dat set was generated by creating games with Battlesnakes that remove the move that moves themselves back on their own neck and otherwise selects moves randomly. The inputs were  the input feature sets for states in the created games. The targets are in WDL-space. By WDL-space, we mean loss=0.0, draw=0.5, and win=1.0. For solo games, the targets are the lesser of the number of moves divided by 100 and 1.0. The efficiently updatable neural networks output a prediction for each move. Gradients were calculated only for the actions played in the states in the created games. Efficiently updatable neural networks for Nuppeppou that are trained with reinforcement learning methods and a mixture of supervised and reinforcement learning methods are works in progress.

On every turn of each game, Nuppeppou first removes moves that move Nuppeppou back on its own neck, hit walls, hit itself, and collide with others from possibility. If all moves are removed from possibility, then Nuppeppou moves up. Otherwise, if an efficiently updatable neural network is assigned to Nuppeppou for this game, then Nuppeppou uses the the efficiently updatable neural network to get scores for each move and selects the move assigned the greatest score and not removed from possibility. Otherwise, Nuppeppou uses flood fill to assign each move not removed from possibility a score. The score is calculated by first calculating the the enclosed space a move moves towards, then assigning all squares in the enclosed space points: empty squares are assigned 1 point, hazard squares are assigned 1/16 points, and food squares are assigned the amount of health Nuppeppou would restore if it consumed food now. The score is the sum of all points. Nuppeppou then selects the move with the greatest score.

On the end of each game, Nuppeppou deallocates some server-side resources. In particular, Nuppeppou deallocates memory of the previous state that was needed for updating accumulators in efficiently updatable neural networks. Nuppeppou does not deallocate efficiently updatable neural networks, but moves them to a stack of  efficiently updatable neural networks to be recycled.
