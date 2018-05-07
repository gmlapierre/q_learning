import numpy as np
import random
import time

# Based on the example at URL http://people.revoledu.com/kardi/tutorial/ReinforcementLearning/Q-Learning-Example.htm

# Create reward matrix
R = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 100],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100],
              [-1, 0, -1, -1, 0, 100]], np.int32)


def get_possible_states(reward_matrix, state):
    # Get line of the state
    possible_states = []
    iterator = 0
    for s in reward_matrix[state]:
        if s != -1:
            possible_states.append(iterator)
        iterator += 1
    return possible_states


def get_random_state(reward_matrix):
    return random.randint(0, reward_matrix.shape[0] - 1)


def initialize_q_matrix(reward_matrix):
    return np.zeros((reward_matrix.shape[0], reward_matrix.shape[1]))


def get_max_q_value(q_matrix, row):
    tmp = q_matrix[row]
    a = np.argwhere(tmp == np.max(tmp))
    if a.size > 1:
        flatten = a.flatten()
        max_q_value = random.choice(flatten)
    else:
        max_q_value = a.flatten()[0]
    return tmp[max_q_value]


def q_learning_algorithm_step_by_step(reward_matrix, gamma, episodes, goal_state):
    print("Starting Q Learning")
    print("Gamma: ", gamma, "Episodes: ", len(episodes), "Goal State: ", goal_state)
    q_matrix = initialize_q_matrix(reward_matrix)
    state_sequence = []
    for _ in episodes:
        random_state = get_random_state(reward_matrix)
        state_sequence.append(random_state)
        print("Initial Random State: ", random_state)
        while True:  # This simulates a do while loop
            next_state = random.choice(get_possible_states(reward_matrix, random_state))
            print("Next State: ", next_state)
            max_q_value = get_max_q_value(q_matrix, next_state)
            print("Max Q Value: ", max_q_value)
            print("Calculating Q Value")
            print("Q[state, action] = R[state, action] + (gamma * max_q_value)")
            print('q_matrix[', random_state, ',', next_state, '] = reward_matrix[', random_state, ',', next_state, ']', '+ (', gamma, ' * ', max_q_value, ')')
            print('q_matrix[', random_state, ',', next_state, '] = ', reward_matrix[random_state, next_state], ' + ', gamma * max_q_value)
            q_matrix[random_state, next_state] = reward_matrix[random_state, next_state] + (gamma * max_q_value)
            print('q_matrix[', random_state, ',', next_state, ']  = ', q_matrix[random_state, next_state])

            random_state = next_state
            state_sequence.append(random_state)
            print('Setting next state as: ', random_state)
            print('Q matrix: ', '\n', q_matrix)

            input("Presse Enter to continue \n")
            if random_state == goal_state:
                print('Next state is the goal state so break')
                print('End of episode \n')
                print('Sequence of states was: ', state_sequence)
                state_sequence.clear()
                break
        input("Presse Enter to continue \n")
    return q_matrix


def q_learning_algorithm(reward_matrix, gamma, episodes, goal_state):
    q_matrix = initialize_q_matrix(reward_matrix)
    state_sequence = []
    for _ in episodes:
        random_state = get_random_state(reward_matrix)
        state_sequence.append(random_state)
        while True:  # This simulates a do while loop
            next_state = random.choice(get_possible_states(reward_matrix, random_state))
            max_q_value = get_max_q_value(q_matrix, next_state)
            q_matrix[random_state, next_state] = reward_matrix[random_state, next_state] + (gamma * max_q_value)

            random_state = next_state
            state_sequence.append(random_state)

            if random_state == goal_state:
                state_sequence.clear()
                break
    return q_matrix


def use_q_matrix(q_matrix, initial_state, goal):
    state_sequence = []
    current_state = initial_state
    # From current state, find the action with the highest Q value
    state_sequence.append(current_state)
    while current_state != goal:
        print('current_state: ', current_state)
        tmp = q_matrix[current_state]
        next_state = (np.argwhere(tmp == np.max(tmp))).flatten()[0]
        state_sequence.append(next_state)
        current_state = next_state
    print(state_sequence)


Q = q_learning_algorithm_step_by_step(R, 0.8, range(100), 5)

use_q_matrix(Q, 0, 5)


print(np.__version__)