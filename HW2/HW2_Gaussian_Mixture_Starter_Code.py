#Generating Gaussian Mixture

import random
import numpy as np
p_head = 0.7    #Prob. of heads
N = 100         #Total Draws

def flip(p):
    return 0 if random.random() < (1 - p_head) else 1

flips = [flip(p) for i in range(N)]

#float(flips.count('H'))/N            # Just a Check to see that we are drawing approx p% heads


#Idea - If Heads Draw from Class(1) or else from Class(0)
def draw(coin):
    if coin == 1:
        return np.random.normal(1.5, 1, 1)[0]               #Mean 1.5, SD 1
    else:
        return np.random.normal(0, 1, 1)[0]                 #Mean 0, SD 1

draws = [draw(toss) for toss in flips]                      #Draws contains the Gaussian Mixture of Points

# Use draws as your "x" variable and flips as your "y" variable
x = draws
y = flips
