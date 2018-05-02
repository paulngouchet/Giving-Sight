import numpy as np
import random
import time
import gym
from gym import wrappers

def run_episode(env, to_be_rendered, policy, episode_len=100): # Playing the entire game once
    total_reward = 0
    obs = env.reset()
    for t in range(episode_len):
        if to_be_rendered == 1 :
            env.render()
        action = policy[obs]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            #print('Epside finished after {} timesteps.'.format(t+1))
            break
    return total_reward


def evaluate_policy(env, policy, n_episodes=100):       # returns the total reward after playing the game
    total_rewards = 0.0
    #for _ in range(n_episodes):
        #total_rewards += run_episode(env, policy)        # One way to find the total by letting the same sample of the population play
                                                          # playing the game multiple times and getting the average of the rewards at the end of all the games
    #return total_rewards/n_episodes
    total_rewards = run_episode(env, 0 , policy)              # Simple reward calculation, the game is played once and OpenAI gym returns the total reward
    return  total_rewards

def gen_random_policy():
    return np.random.choice(4, size=((16)))             # generating one member of the population/ policy

def crossover(policy1, policy2):                        # Doing a crossover with a random probability where genes of two members can be exchanged to give an offspring
    new_policy = policy1.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand > 0.5:
            new_policy[i] = policy2[i]
    return new_policy
                                    # Other way to create an offspring where we just split the list in 2 and compose the offspring the first half of the first parent
                                    # and the second half of the other parent
    '''
        length = int(len(policy1) / 2)
        first_part = policy1[:length]
        second_part = policy2[length:]

        breeding = np.append(first_part, second_part)

        return breeding'''

def mutation(policy, p=0.05):                           # Doing a mutation a with a certain probability of happening
    new_policy = policy.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand < p:
            new_policy[i] = np.random.choice(4)
    return new_policy

# OpenAI gym is a toolking for testing reinforcement learning algorithm where an agent goes through a game and get points ( rewards )
# For every actions that influences the environment taken until the end. At the end the total number of points is calculated and it is called the maximum reward
# It is a tool to test reinforcement learning algorithm

# The goal of this Project is to combine Genetic Algorithm with reinforcement learning
# In Genetic you have one main goal find the best policy which is the action at each state to maximize the total_reward
# In this code i am applying my code to the FrozenLake example Where you need to reach a Goal from the start to end with 16 possible states
# At each state you can either go UP, DOWN, LEFT OR RIGHT.
# In this game you score 0 point for every actions until you reach the goal and you score 1 point
# The Goal of our genetic is to find the optimal policy where an element of the population a list of the 16 possible state where each index
# Corresponds to an action, either going up, down, left or right.
# Example of an element of the population. [0, 1, 2, 2, 1, 3, 0, 2, 0, 1,3, 2, 3, 1, 2, 0]
# The fitness function is the maximum reward after going through a episode of the OpenAI Gym

if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    env = gym.make('FrozenLake-v0')    #Uploading FrozenLake environment from OpenAI
    env.seed(0)
    env = wrappers.Monitor(env, '/tmp/frozenlake1', force=True)
    ## Policy search
    n_policy = 50      # Generating 50 different policy/ a population of 16 elements
    n_steps = 20       # Number of generation of the genetic algorithm before stopping
    start = time.time()
    policy_pop = [gen_random_policy() for _ in range(n_policy)]  #Generating the policy and building a 2d array of the policies
    for idx in range(n_steps):
        policy_scores = [evaluate_policy(env, p) for p in policy_pop] # Evaluation each policy and giving a score
        print('Generation %d : max score = %0.2f' %(idx+1, max(policy_scores))) # The maximum reward obtained by the best policy at this generation
        policy_ranks = list(reversed(np.argsort(policy_scores)))     # Ranking the policy by their index

        elite_set = [policy_pop[x] for x in policy_ranks[:5]]       # Accessing the five best policies

        select_probs = np.array(policy_scores) / np.sum(policy_scores)  # Giving a weight to each member of the population for breeding

        child_set = [crossover(
            policy_pop[np.random.choice(range(n_policy), p=select_probs)],  # Creating offsprings
            policy_pop[np.random.choice(range(n_policy), p=select_probs)])  # Example of Offspring
            for _ in range(n_policy - 5)]
            # [0, 1, 2, 2, 1, 3, 0, 2,|  0, 1,3, 2, 3, 1, 2, 0] , [3, 1, 2, 3, 1, 3, 2, 2,| 0, 1, 3, 1, 2, 1, 2, 1]
            # ==> [0, 1, 2, 2, 1, 3, 0, 2,| 0, 1, 3, 1, 2, 1, 2, 1]
        mutated_list = [mutation(p) for p in child_set]  # Creating a random mutation with a certain probability by changing randomly of the moves at a state
        policy_pop = elite_set # Keeping the five best policies from the previous
        policy_pop += mutated_list # Appending the offsprings of 15 samples breed given their weight based on their score
    policy_score = [evaluate_policy(env, p) for p in policy_pop] # End of the genetic algorithm with all the policies
    best_policy = policy_pop[np.argmax(policy_score)] # select the best mutation / policy for the game

    end = time.time()
    #print('Best policy is ')
    #print(best_policy)
    print('Best policy score = %0.2f. Time taken = %4.4f'
            %(np.max(policy_score), (end-start)))

    ## Evaluation
    #env = wrappers.Monitor(env, '/tmp/frozenlake1', force=True)
    for _ in range(10):
        print(" End of the game ")
        run_episode(env,1, best_policy) # Running the game 10 times  with the best policy
    print('Best policy score = %0.2f. Time taken = %4.4f'
            %(np.max(policy_score), (end-start)))
    env.close()
    #gym.upload('/tmp/frozenlake1', api_key=...)
