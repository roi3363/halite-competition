from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import tensorflow as tf
import numpy as np
import constants

# Create a test environment for use later
agent_count = 2
environment.reset(agent_count)


def train_ship(
    number_training_examples,
    number_test_examples,    
    neighborhood_shape=(5, 5),
    board_size=5,
    num_actions=3
):
    train_neighborhoods, train_actions = generate_examples(
        number_training_examples,
        neighborhood_shape,
        board_size,
        num_actions
    )

    test_neighborhoods, test_actions = generate_examples(
        number_test_examples,
        neighborhood_shape,
        board_size,
        num_actions
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=neighborhood_shape),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_actions * 5, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(train_neighborhoods, train_actions, epochs=10)

    test_loss, test_acc = model.evaluate(test_neighborhoods,  test_actions, verbose=2)

    print('\nTest accuracy:', test_acc)
    
    return model


def generate_examples(
    num_examples,
    neighborhood_shape=(5, 5),
    board_size=5,
    num_actions=3
):
    train_neighborhoods = []
    train_actions = []
    for i in range(num_examples):
        environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
        environment.reset(agent_count)
        state = environment.state[0]
        board = Board(state.observation, environment.configuration)
        ship = board.current_player.ships[0]
        ship_neighborhood = get_ship_neighborhood(ship, board, neighborhood_shape)
        ship_action = get_ship_action(ship, board, num_actions)

        train_neighborhoods.append(ship_neighborhood)
        train_actions.append(ship_action)

    return np.array(train_neighborhoods), np.array(train_actions)
        
def get_ship_action(ship, board, num_actions=3):
    """returns ship action for num of actions as an np array"""
    neighborhood_size = 2
    neighborhood_shape = (2 * neighborhood_size + 1, 2 * neighborhood_size + 1)
    neighborhood_values = get_halite_cell_value(
        ship,
        board,
        neighborhood_shape,
        3
    )

    highest_val_pos = np.argmax(neighborhood_values)
    highest_val_indices = np.unravel_index(highest_val_pos, neighborhood_shape)
    ship_origin = get_ship_origin(ship, neighborhood_shape).reshape(2)

    moves_indices = np.array(highest_val_indices) - ship_origin

    horizontal_moves = move_index_to_arr(moves_indices[0], constants.HORIZONTAL)
    vertical_moves = move_index_to_arr(moves_indices[1], constants.VERTICAL)
    moves = np.append(horizontal_moves, vertical_moves)

    if np.equal(horizontal_moves, constants.HOLD).all():
        moves = np.append(vertical_moves, horizontal_moves)
    
    num_zeros = num_actions * 5 - moves.shape[0]

    if num_zeros > 0:
        hold_moves = np.zeros(num_zeros)
        moves = np.append(moves, hold_moves)

    return moves



def move_index_to_arr(index, direction):
    if index == 0:
        return constants.HOLD

    moves_array = []
    if direction == constants.HORIZONTAL:
        if index < 0:
            moves_array = [constants.WEST]

        else:
            moves_array = [constants.EAST]
    
    else:
        if index < 0:
            moves_array = [constants.NORTH]
        else:
            moves_array = [constants.SOUTH]

    return np.array(abs(index) * moves_array)


def get_halite_cell_value(
    ship,
    board,
    neighborhood_shape,
    num_actions
) -> np.array:
    """returns array of values representing how worth it is to move to that cell"""
    halite_neigborhood = get_ship_neighborhood(ship, board, neighborhood_shape)
    neighborhood_indices = get_ship_centered_indices(
        ship,
        board,
        neighborhood_shape
    )

    def distance(cell_indices):
        return sum(abs(cell_indices))
    
    neighborhood_distances = np.apply_along_axis(
        distance,
        axis=0,
        arr=neighborhood_indices
    )

    def cell_value_ratio(distance):
        ratio = 0
        for i in range(num_actions - distance):
            ratio += (1 / 4) ** (i + 1)

        return ratio
    
    vec_cell_value_ratio = np.vectorize(cell_value_ratio, otypes=[float])

    return halite_neigborhood * vec_cell_value_ratio(neighborhood_distances)


def get_ship_origin(ship, neighborhood_shape):
    size = (neighborhood_shape[0] - 1) // 2
    return np.array([[size, size]]).reshape(2, 1, 1)


def get_ship_centered_indices(ship, board, neighborhood_shape) -> np.array:
    """returns array reprsenting distance from center (ship position)"""
    neighborhood_indices = np.indices(neighborhood_shape)
    neighborhood_origin = get_ship_origin(ship, neighborhood_shape)
    neighborhood_indices -= neighborhood_origin

    return neighborhood_indices


def get_ship_neighborhood(ship, board, shape=(5, 5)) -> np.array:
    """
    Returns square array represention of halite board centered at ship with 
    size the number of block to the left, right, up and down
    """
    board_size = board.configuration.size

    if board_size < shape[0]:
        print("size of neighborhood is greater than size of board")

    neighborhood_indices = get_ship_centered_indices(ship, board, shape)
    board_indices = (
        neighborhood_indices + np.array(ship.position).reshape(2, 1, 1)
    ) % board_size

    def get_halite_at_cell(cell_indices):
        return board.cells[tuple(cell_indices)].halite

    halite_neigborhood = np.apply_along_axis(
        get_halite_at_cell,
        axis=0,
        arr=board_indices
    )

    return halite_neigborhood
