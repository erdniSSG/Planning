import numpy as np
import numpy.random as rand
from datetime import datetime
import time
import random
import sys

def load_mdp(filename, g):
    file = np.load(filename)
    X = file["X"]
    A = file["A"]
    P = file["P"]
    c = file["c"]
    P_a = tuple(P[i] for i in range(0, len(A)))
    tup = (X, A, P_a, c, g)
    return tup

print(load_mdp("taxi.npz", 0.99))


def noisy_policy(tup, a, eps):
    
    pi = np.ones((len(tup[0]), len(tup[1])), dtype="float")
    
    div = eps/(len(tup[1])-1)
    
    pi *= div
    
    for i in pi:
        i[a] = (1-eps)
      
    return pi

print(noisy_policy(load_mdp("taxi.npz", 0.99), 0, 0.8))

#policy evaluation
def evaluate_pol(tup, pi):
    
    P_pi = np.zeros((len(tup[0]), len(tup[0])), dtype="float")
    c_pi = np.zeros((len(tup[0]), 1), dtype="float")
    
    
    for i in range(0, len(P_pi)):
        for j in range(0, len(tup[1])):
            P_pi[i] += pi[i][j]*tup[2][j][i]
            c_pi[i] += pi[i][j]*tup[3][i][j]
    

    I = np.identity(len(P_pi))
    
    J_pi = np.dot(np.linalg.inv(I-tup[4]*P_pi), c_pi)
    
    return J_pi

print(evaluate_pol(load_mdp("taxi.npz", 0.99), noisy_policy(load_mdp("taxi.npz", 0.99), 0, 0.8)))


def value_iteration(tup):
 
    starttime = time.time()

    X = tup[0]
    A = tup[1]
    P = tup[2]
    c = tup[3]
    gamma = tup[4]
    eps = 0.00000001
    
    #initialize J with the size of state-space
    J = np.zeros((len(X), 1))
    err = 1.0
    niter = 0
    
    while err > eps:
        #auxiliary array to store intermediate values
        Q = np.zeros((len(X), len(A)))
        
        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)
            
        #compute minimum row-wise
        Jnew = np.min(Q, axis=1, keepdims=True)
        
        #compute error
        err = np.linalg.norm(J - Jnew)
        
        #update
        J = Jnew
        niter += 1
    
    J_pi = evaluate_pol(load_mdp("taxi.npz", 0.99), noisy_policy(load_mdp("taxi.npz", 0.99), 0, 0.8))
    
    print('- Is the policy from activity 3 optimal?', np.all(np.isclose(J, J_pi)))

    endtime = time.time()
    
    t = np.round((endtime - starttime), 3)
    
    print(f'Execution time: {t}' " seconds ")
    print(f'N. iterations: {niter}')
    
    
    return np.round(J, 3)

print(value_iteration(load_mdp("taxi.npz", 0.99)))


def policy_iteration(tup):
    
    starttime = time.time()
    
    X = tup[0]
    A = tup[1]
    P = tup[2]
    c = tup[3]
    gamma = tup[4]

    #initialize pi with the uniform policy
    pol = np.ones((len(X), len(A))) / len(A)
    quit = False
    niter = 0

    while not quit:
        #auxiliary array to store intermediate values
        Q = np.zeros((len(X), len(A)))

        #policy evaluation
        cpi = np.sum(c * pol, axis=1, keepdims=True)
        Ppi = np.zeros((len(X), len(X)))
        for a in range(len(A)):
            Ppi += pol[:, a, None] * P[a]
        J = np.linalg.inv(np.eye(len(X)) - gamma * Ppi).dot(cpi)

        #compute Q-values
        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)

        #compute greedy policy
        pmin = np.argmin(Q, axis=1)
        pnew = np.eye(len(A))[pmin]

        #compute stopping condition
        quit = (pol == pnew).all()

        #updating
        pol = pnew
        niter += 1


    
    endtime = time.time()
    
    t = np.round((endtime - starttime), 3)

    print("Yes, the policy is optimal")
    print(f'Execution time: {t}' " seconds ")
    print(f'N. iterations: {niter}')
    
    
    
    return np.round(pol, 3)

print(policy_iteration(load_mdp("taxi.npz", 0.99)))


NRUNS = 100 

def simulate(tup, pol, x0, length):
    
    x = x0
    states = np.arange(len(tup[0]))
    actions = np.arange(len(tup[1]))
    totcost = np.zeros(NRUNS)
    
    
    for i in range(0, NRUNS):
        cost = 0
        for j in range(0, length):
            chosen_action = random.choices(actions, weights=pol[x], k=1)[0]
            cost += tup[4]**j*tup[3][x][chosen_action]
            x = random.choices(states, weights=tup[2][chosen_action][x], k=1)[0]
        totcost[i] = cost   
        
         
    avcost = sum(totcost)/NRUNS
    
    return avcost, totcost
    
#simulation
print(simulate(load_mdp("taxi.npz", 0.99), policy_iteration(load_mdp("taxi.npz", 0.99)), 17, 1000))
    




