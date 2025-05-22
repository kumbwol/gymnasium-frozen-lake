from learningLoop import LearningLoop
import gymnasium

from playTest import PlayTest

env=gymnasium.make('FrozenLake-v1')
learn = LearningLoop(env)
print("learning finished")
env2=gymnasium.make('FrozenLake-v1', render_mode = "human")
PlayTest(env2, learn.agent)

'''

import gymnasium


env=gymnasium.make('FrozenLake-v1')
env.reset()
#env.render()

numOfIterations = 1000000 # random lepessel - 1803

winAmount = 0

for i in range(numOfIterations):
    randomAction = env.action_space.sample()
    returnValue = env.step(randomAction)
    #print(returnValue, i)


    if returnValue[2]:
        if returnValue[0] == 15:
            winAmount+=1
        env.reset()

print("final", winAmount)
env.close()
'''