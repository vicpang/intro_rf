#import an env
import gym
env=gym.make('CartPole-v0')

#reset an env
env.reset()

for _ in range(1000):
	env.render()
	#sample an action
	action=env.action_space.sample()
	#take an action
	observation, reward, done, info=env.step(action)
env.close()