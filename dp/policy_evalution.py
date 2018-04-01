# set up env
import numpy as np 
import pprint
import sys
if "../" not in sys.path:
	sys.path.append("../")
from lib.envs.gridworld import GridworldEnv 
env = GridworldEnv()
pp = pprint.PrettyPrinter(indent=2)


# evaluate policy of env 
def policy_eval(policy, env, discount=1.0,theta=0.0001):
	V=np.zeros(env.nS)
	converged=False
	#delta=0
	while not converged:
		delta=0
		for s in range(env.nS):
			v=0
			for a,action_prob in enumerate(policy[s]):
				for prob, next_state, reward, done in env.P[s][a]:
					v+=action_prob*prob*(reward+discount*V[next_state])
			
			delta=max(delta,np.abs(v-V[s])) 
			V[s]=v
		print(delta)
		if delta<theta:
			converged=True
	return np.array(V)

#init a random policy and evaluate that policy in our env


random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")