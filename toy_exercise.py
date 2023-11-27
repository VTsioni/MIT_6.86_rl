import numpy as np
gamma = 0.5
states = np.arange(0, 5)  # a grid of 5 states [][][][][]
actions = np.arange(0, 3)  # like the choices we have stay move left, move right
rewards = np.zeros(5)
rewards[4] = 1 #if states == 4 else 0  # for state 5 0 for others

transition_matrix = np.zeros((len(states), len(actions), len(states)))

choice = {"stay": 0, "move_left": 1, "move_right": 2}

def trans_mat(choice):
    if choice == 0:   # stay
        for state in states:
            transition_matrix[state, choice, state] = 0.5  # [0,0,0]=[1,0,1]=[2,0,2]=[3,0,3]=[4,0,4]=0.5
            if state == 0:
                transition_matrix[state, choice, (state + 1)] = 0.5  # [0,0,1]=0.5
            elif state == 4:
                transition_matrix[state, choice, (state - 1)] = 0.5  # [4,0,3]=0.5
            else:
                transition_matrix[state, choice, (state + 1)] = 0.25  # [1,0,2]=[2,0,3]=[3,0,4]=0.25
                transition_matrix[state, choice, (state - 1)] = 0.25  # [1,0,0]=[2,0,1]=[3,0,2]=0.25
    else:  # moves either left or right
        if choice == 1:  # moves left
            for state in np.arange(1, 5):
                transition_matrix[state, choice, (state-1)] = 1.0 / 3  # actually moves
                transition_matrix[state, choice, state] = 2.0 / 3  # stays

            transition_matrix[0, choice, 0] = 0.5
            transition_matrix[0, choice, 1] = 0.5
        if choice == 2:  # moves right
            for state in np.arange(0, 4):
                transition_matrix[state, choice, (state+1)] = 1.0 / 3  # actually moves
                transition_matrix[state, choice, state] = 2.0 / 3  # stays

            transition_matrix[4, choice, 4] = 0.5
            transition_matrix[4, choice, 3] = 0.5
    return transition_matrix

v_value = np.zeros(5)
num_iterations = 100
choices = []
for iter in range(num_iterations):
    choice = np.random.randint(0, 3)
    choices.append(choice)
    transition_matrix = trans_mat(choice)
    q_value = np.dot(transition_matrix, (rewards + gamma * v_value))
    v_value = np.max(q_value, axis=1)
    print(v_value)
    print(choices)

