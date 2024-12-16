import random
import numpy as np
import sys
import numpy.random as rand

def load_pomdp(filename, g):
    file = np.load(filename)
    X = file["X"]
    A = file["A"]
    Z = file["Z"]
    P = file["P"]
    O = file["O"]
    c = file["c"]
    P_a = tuple(P[i] for i in range(0, len(A)))
    O_a = tuple(O[i] for i in range(0, len(A)))
    tup = (X, A, Z, P_a, O_a, c, g)
    return tup

print(load_pomdp("pomdp.npz", 0.99))

M = load_pomdp('pomdp.npz', 0.99)

rand.seed(42)

#states
print('= State space (%i states) =' % len(M[0]))
print('\nStates:')
for i in range(len(M[0])):
    print(M[0][i]) 

#random state
x = rand.randint(len(M[0]))
print('\nRandom state: x =', M[0][x])

#last state
print('\nLast state:', M[0][-1])

#actions
print('= Action space (%i actions) =' % len(M[1]))
for i in range(len(M[1])):
    print(M[1][i]) 

#random action
a = rand.randint(len(M[1]))

print('\nRandom action: a =', M[1][a])

#observations
print('= Observation space (%i observations) =' % len(M[2]))
print('\nObservations:')
for i in range(len(M[2])):
    print(M[2][i]) 

#random observation
z = rand.randint(len(M[2]))
print('\nRandom observation: z =', M[2][z])

#last state
print('\nLast observation:', M[2][-1])

#transition probabilities
print('\n= Transition probabilities =')

for i in range(len(M[1])):
    print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[3][i].shape)
    print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[3][i]), len(M[0])))
    
print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][x], M[1][a]))
print("s' in", np.array(M[0])[np.where(M[3][a][x, :] > 0)])

#observation probabilities
print('\n= Observation probabilities =')


for i in range(len(M[1])):
    print('\nObservation probability matrix dimensions (action %s):' % M[1][i], M[4][i].shape)
    print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[4][i]), len(M[0])))
    
print('\nState-action pair (%s, %s) yields observation(s)' % (M[0][x], M[1][a]))
print("z in", np.array(M[2])[np.where(M[4][a][x, :] > 0)])

#cost
print('\n= Costs =')

print('\nCost for the state-action pair (%s, %s):' % (M[0][x], M[1][a]))
print('c(s, a) =', M[5][x, a])

#discount
print('\n= Discount =')
print('\ngamma =', M[6])

def gen_trajectory(tup, x0, n):
    
    x_arr = np.zeros(n+1)
    x_arr[0] = x0
    a_arr = np.zeros(n)
    o_arr = np.zeros(n)
    curr_x = x0
    states = list(range(0, len(tup[0]), 1))
    actions = list(range(0, len(tup[1]), 1))
    obs = list(range(0, len(tup[2]), 1))
    
    for i in range(1, n+1):
        a_arr[i-1] = np.random.choice(actions)
        curr_x = np.random.choice(states, p=tup[3][int(a_arr[i-1])][curr_x])
        x_arr[i] = curr_x
        o_arr[i-1] = np.random.choice(obs, p=tup[4][int(a_arr[i-1])][curr_x])
        
    tup_2 = (x_arr, a_arr, o_arr)
    
    return tup_2


print(gen_trajectory(load_pomdp("pomdp.npz", 0.99), 0, 10))

def sample_beliefs(tup, n):
    
    states = list(range(0, len(tup[0]), 1))
    traj = gen_trajectory(tup, np.random.choice(states), n)
       
    beliefs = np.zeros((n+1, len(tup[0])))
    
    beliefs[0] = np.ones(len(tup[0]))*1/len(tup[0])
    
    for i in range(1,n+1):
        observation = tup[4][(int(traj[1][i-1]))].T
        beliefs[i] = np.dot(np.dot(beliefs[i-1], tup[3][int(traj[1][i-1])]), np.diag(observation[int(traj[2][i-1])])) * 1 / np.linalg.norm(np.dot(np.dot(beliefs[i-1], tup[3][int(traj[1][i-1])]), np.diag(observation[int(traj[2][i-1])])))

    return beliefs

print(sample_beliefs(load_pomdp("pomdp.npz", 0.99), 3))

def solve_mdp(tup):

    #these variables are just to help readability
    X = tup[0]
    A = tup[1]
    P = tup[3]
    c = tup[5]
    gamma = tup[6]
    eps = 0.00000001
    
    #initialize J with the size of state-space
    Q_vec = np.zeros((len(X), 1))
    err = 1.0
    niter = 0
    
    while err > eps:
        #auxiliary array to store intermediate values
        Q = np.zeros((len(X), len(A)))
        
        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(Q_vec)
            
        #compute minimum row-wise
        Q_vec_new = np.min(Q, axis=1, keepdims=True)
        
        #compute error
        err = np.linalg.norm(Q_vec - Q_vec_new)
        
        #update
        Q_vec = Q_vec_new
        niter += 1
    
    
    return Q

print(solve_mdp(load_pomdp("pomdp.npz", 0.99)))

def get_heuristic_action(beliefs, Q_func, heuristic):
    """
    Returns the action index based on the given heuristic.
    
    :param beliefs: A numpy array representing the belief state.
    :param Q_func: A numpy array representing the optimal Q-function for an MDP.
    :param heuristic: A string that can be 'mls', 'av', or 'q-mdp'.
    :return: An integer corresponding to the index of the action prescribed by the heuristic.
    """
    
    if heuristic == 'mls':
        #most likely state heuristic
        most_likely_state = np.argmax(beliefs)
        #find the action that has the highest Q value in the most likely state
        action_index = np.argmax(Q_func[most_likely_state, :])
        
    elif heuristic == 'av':
        #action Voting heuristic
        #for each action, sum the product of belief and Q values across all states
        action_values = np.sum(beliefs[:, np.newaxis] * Q_func, axis=0)
        #the best action is the one with the highest total value
        action_index = np.argmax(action_values)
        
    elif heuristic == 'q-mdp':
        #Q-MDP heuristic
        #compute the expected Q value for each action by weighting with the belief state
        expected_Q_values = np.sum(beliefs[:, np.newaxis] * Q_func, axis=0)
        #the best action is the one with the highest expected Q value
        action_index = np.argmax(expected_Q_values)
        
    else:
        raise ValueError("Invalid heuristic. Choose 'mls', 'av', or 'q-mdp'.")
    
    return action_index

#example usage
belief_state = np.array([0.1, 0.4, 0.5])  #dummy belief state
Q_function = np.array([[10, 0], [0, 20], [5, 15]])  #dummy Q-function
heuristic_choice = 'mls'  

print(get_heuristic_action(belief_state, Q_function, heuristic_choice))

def solve_fib(pomdp):
    max_iterations=1000
    tolerance = 1e-1
    states, actions, observations, transition_prob, observation_prob, cost, gamma = pomdp
    num_states = len(states)
    num_actions = len(actions)
    
    #initialize Q function arbitrarily for all state-action pairs
    Q_fib = np.zeros((num_states, num_actions))
    Q_fib_new = np.copy(Q_fib)
    
    iteration = 0
    error = np.inf
    
    #iterating until the error is smaller than the tolerance or we reach max iterations
    while error > tolerance and iteration < max_iterations:
        #compute the Q function for each state-action pair
        for i, state in enumerate(states):
            for j, action in enumerate(actions):
                sum_over_next_states = 0
                for k, next_state in enumerate(states):
                    #calculate the minimum Q for the next state over all actions
                    min_Q_next_state = min(Q_fib[k, :])
                    #expected observation value over all possible observations
                    expected_observation_value = sum(
                        observation_prob[next_state, action, l] * min_Q_next_state
                        for l in range(len(observations))
                    )
                    sum_over_next_states += transition_prob[state, action, k] * expected_observation_value
                
                #update the Q value for the current state-action pair
                Q_fib_new[i, j] = cost[state, action] + gamma * sum_over_next_states
        
        #calculate the maximum difference between the old and new Q-values for the stopping criterion
        error = np.max(np.abs(Q_fib_new - Q_fib))
        
        #update Q_fib with the new values for the next iteration
        Q_fib = np.copy(Q_fib_new)
        
        iteration += 1
    
    return Q_fib, iteration

#dummy POMDP components for testing the function
states = np.arange(3)
actions = np.arange(2)
observations = np.arange(2)
transition_prob = np.random.rand(len(states), len(actions), len(states))
observation_prob = np.random.rand(len(states), len(actions), len(observations))
cost = np.random.rand(len(states), len(actions))
gamma = 0.9

#POMDP tuple
pomdp = (states, actions, observations, transition_prob, observation_prob, cost, gamma)

#test
Q_fib_result, iterations = solve_fib(pomdp)
Q_fib_result, iterations
