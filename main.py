import gymnasium
import time

env=gymnasium.make('FrozenLake-v1', render_mode = "human")
env.reset()
env.render()

numOfIterations = 300

for i in range(numOfIterations):
    randomAction = env.action_space.sample()
    returnValue = env.step(randomAction)
    print(returnValue, i)


    if returnValue[2]:
        env.reset()

env.close()