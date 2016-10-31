import blackjack
from pylab import *
import numpy as np

Q1 = ... # NumPy array of correct size
Q2 = ... # NumPy array of correct size

def learn(alpha, eps, numTrainingEpisodes):
    ... # Fill in Q1 and Q2

def evaluate(numEvaluationEpisodes):
    returnSum = 0.0
    for episodeNum in range(numEvaluationEpisodes):
        G = 0
        ...
        returnSum = returnSum + G
    return returnSum/numEvaluationEpisodes

def main(epsilon, numEpisodes):
    global Q1, Q2
    Q1=0.0000001*np.random.random((181,2)) # initialize the estimates
    Q2=0.0000001*np.random.random((181,2)) # initialize the estimates

    returnSum = 0.0    
    alpha=0.001
    gamma = 1.0
    for episodeNum in range(numEpisodes):
        G = 0
        state = blackjack.init()
        while(True):
            action = epsilonGreedyAction(state, epsilon)
            reward,nextState = blackjack.sample(state,action)
            if np.random.randint(0,2): # with 0.5 probability
                Q1[state][action]=Q1[state][action]+alpha*(reward+gamma*Q2[nextState][argmax(Q1[nextState])]-Q1[state][action])
            else:  # with 0.5 probability
                Q2[state][action]=Q2[state][action]+alpha*(reward+gamma*Q1[nextState][argmax(Q2[nextState])]-Q2[state][action])
            state = nextState
            G = G + reward # update the return for state 0 with discount ratio gamma=1
                
            if state==False:
                break
        returnSum = returnSum + G

        if not episodeNum%10000 and episodeNum!=0:
            #print("return sum is: ", returnSum)
            #print("episode number is: ", episodeNum)
            print ("Average return after %d episodes: %f"%(episodeNum, returnSum/episodeNum))
    print ("Average return after all episodes: %f"%(numEpisodes, returnSum/episodeNum))

    # After learning, 
    # choose deterministic policy greedy with respect to the sum of estimate values
    deterministicSum=0.0
    for episodeNum in range(numEpisodes):
        G = 0
        state=blackjack.init()
        while(True):
            # choose the action greedy wrt the sum of two action values
            reward,state=blackjack.sample(state, argmax(Q1[state]+Q2[state]))
            G=G+reward

            if state==False:
                break
        deterministicSum = deterministicSum + G
    print ("determinstic return after learning: ",deterministicSum/numEpisodes)
    blackjack.printPolicy(greedyAction)

def epsilonGreedyAction(state, epsilon):
    num=np.random.random()
    if (num>=epsilon):
        return argmax(Q1[state]+Q2[state])
    else:
        return (np.random.randint(0,2))

def greedyAction(state):
    '''
    choose the action greedy wrt the sum of two action values
    '''
    return argmax(Q1[state]+Q2[state])

if __name__ == "__main__":
    #epsilon = 1
    #main(epsilon, 10000)
    epsilon = 0.01
    main(epsilon, 1000000)