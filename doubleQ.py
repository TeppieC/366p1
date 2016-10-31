import blackjack
from pylab import *
from numpy import *


def main(epsilon, numEpisodes):
    repnums=100
    global Q1, Q2
    Q1=0.0000001*rand(181,2) # initialize the estimates
    Q2=0.0000001*rand(181,2) # initialize the estimates
    for repnum in range(repnums):
        returnSum = 0.0
        for episodeNum in range(numEpisodes):
            G = 0
            stepsize=0.001
            discount=1.0
            state=blackjack.init()
            while(state!=False):
                action=chooseAction(state, epsilon)
                reward,statenext=blackjack.sample(state,action)
                if randint(0,2):
                    Q1[state][action]=Q1[state][action]+stepsize*(reward+discount*Q2[statenext][argmax(Q1[statenext])]-Q1[state][action])
                else:                    
                    Q2[state][action]=Q2[state][action]+stepsize*(reward+discount*Q1[statenext][argmax(Q2[statenext])]-Q2[state][action])
                state=statenext
                G=G+reward
            #print "Episode: ", episodeNum, "Return: ", G
            returnSum = returnSum + G
        #allSum=allSum+G
        print ("Average return every numEpisodes: ", returnSum/numEpisodes)
    #after learining the policy
    final_sum=0.0
    for episodeNum in range(repnums*numEpisodes):
        G = 0
        state=blackjack.init()
        while(state!=False):
            reward,state=blackjack.sample(state,learning(state))
            G=G+reward
        final_sum = final_sum + G
    print ("det return after learning",final_sum/(repnums*numEpisodes))
    blackjack.printPolicy(learning)

def chooseAction(state, epsilon):
    num=random.random()
    if (num>=epsilon):
        return argmax(Q1[state]+Q2[state])
    else:
        return (random.randint(0,2))

def learning(state):
    return argmax(Q[state])

if __name__ == "__main__":
    epsilon = 1
    main(epsilon, 100)
    epsilon = 0.001
    #main(epsilon, 1000000)