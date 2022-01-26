# from mock_manager import SideChannelManager

import uuid
import struct

from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.logging_util import get_logger


import sys
sys.modules['mlagents_envs.side_channel.side_channel_manager'] = __import__('mock_manager')

import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple 


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import KerasModel as Model
# from model_new import NumpyModel as Model

# import sys



def evaluate_fitness(fitnesses, best_number):
  # offset = 1
  # arr = (fitnesses - fitnesses.min() + offset) / (fitnesses.max() - fitnesses.min() + offset)
  # proba = arr / arr.sum()
  # index = np.arange(len(fitnesses))
  # print(proba, 'proba')
  # choice = np.random.choice(index , best_number, p=proba, replace=False)
  # print(np.flip(np.argsort(fitnesses)), fitnesses[np.flip(np.argsort(fitnesses))], fitnesses[np.flip(np.argsort(fitnesses))][best_number:], 'fitnesses[np.argmax(fitnesses)]')
  choice = np.flip(np.argsort(fitnesses))[best_number:]
  return choice
  # return np.flip(np.argsort(fitnesses))

def cross_over(weights_bias, best_ids, population):
  new_weights_bias = []
  while len(new_weights_bias) < population:
    choice = np.random.choice(len(best_ids), 2)
    male_id, female_id = best_ids[choice]

    # crossover weights
    male_weights = weights_bias[male_id][0]
    female_weights = weights_bias[female_id][0]
    new_weights = []
    for i in range(len(male_weights)):
      crossover_choice = np.random.choice([0,1]) 
      if crossover_choice == 0:
        new_weights.append(male_weights[i])
      else:
        new_weights.append(female_weights[i])

      # Mutate weights
      mutation_rate = 0.055
      # mutation_rate = 0
      if np.random.rand() < mutation_rate:
        # print(new_weights[0], 'new_weights')
        rows = new_weights[0].shape[0]
        cols = new_weights[0].shape[1]
        max_random_points = int((rows * cols) / 7)
        random_points = 1
        if max_random_points > 1:
          random_points = np.random.randint(1, max_random_points)

        # mutate weights
        for i in range(random_points):
          random_col = np.random.randint(0, cols)
          random_row = np.random.randint(0, rows)
          new_weights[0][random_row, random_col] = np.clip(new_weights[0][random_row, random_col] + (np.random.rand() * 2 - 1), -1, 1);

        # print(new_weights[0], 'new_weights[0]')
    
    # crossover bias
    male_bias = weights_bias[male_id][1]
    female_bias = weights_bias[female_id][1]
    new_bias = []
    for i in range(len(male_bias)):
      crossover_choice = np.random.choice([0,1]) 
      if crossover_choice == 0:
        new_bias.append(male_bias[i])
      else:
        new_bias.append(female_bias[i])

    new_weights_bias.append([new_weights, new_bias])

  return new_weights_bias


def init_env():
  channel = EngineConfigurationChannel()

  seed = int(1000 * np.random.rand())
  env = UE(file_name='Fish Schooling Simulation 2D', seed=seed, side_channels=[channel])
  # env = UE(file_name='Fish Schooling Simulation 2D', seed=seed)
  
  channel.set_configuration_parameters(time_scale = 5.0)


  env.reset()

  # behavior_name = env.get_behavior_names()[0]
  # behavior_name = "fish?team=0"
  behavior_name = "FishFoodCollector?team=0"

  # spec = env.get_behavior_spec(behavior_name)

  # print("Number of observations : ", len(spec.observation_shapes))

  # if spec.is_action_continuous():
  #   print("The action is continuous")

  # if spec.is_action_discrete():
  #   print("The action is discrete")
  behavior_names = env.behavior_specs.keys()
  for behavior in behavior_names:
    print(behavior, 'behavior')
  return env, behavior_name


def main():
  print('start')
  save_weights = False
  population = 32
  # inputs_shape = 26 + 7
  inputs_shape = 38
  outputs_shape = 2
  # model = Model(population, inputs_shape, outputs_shape, 'model_weights-obstacle-3-hidden/model-84')
  model = Model(population, inputs_shape, outputs_shape)

  env, behavior_name = init_env()

  decision_steps, terminal_steps = env.get_steps(behavior_name)
  
  fitnesses = np.full(population, 0)
  for episode in range(102):
    print(episode, 'episode')
    
    # env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1 # -1 indicates not yet tracking
    done = False # For the tracked_agent

    time_step = 0
    while not done:
      if tracked_agent == -1 and len(decision_steps) >= 1:
        tracked_agent = decision_steps.agent_id[0]
      time_step += 1


      # Set the actions
      # y_positions = []
      if len(decision_steps.agent_id) != 0:
        # for i in range(population):
        #   # print(np.array(decision_steps.obs).shape, 'decision_steps.obs')
        #   observation = decision_steps.obs[0][i]
          # y_positions.append(observation[4])
        # print(decision_steps.obs, 'decision_steps.obs')
        # print(np.array(decision_steps.obs[0][0]), 'decision_steps.obs')
        # print(np.array(decision_steps.obs[1]).shape, 'decision_steps.obs')
        # print(list(decision_steps.obs[0][0]), 'list(decision_steps.obs[0])')
        observation_all = np.concatenate((decision_steps.obs[0], decision_steps.obs[1]), axis = 1)
        action = model.generate_action(list(observation_all))

        # print(behavior_name, action, 'behavior_name, action')
        action_tuple = ActionTuple(continuous = action)
        env.set_actions(behavior_name, action_tuple) 
      # Move the simulation forward
      env.step()
      # Get the new simulation results
      decision_steps, terminal_steps = env.get_steps(behavior_name)


      # finish_order = []
      if len(decision_steps.reward) != 0:
        for i in range(population):
          # print(decision_steps[i], len(decision_steps), 'decision_steps[i]', i)
          # print(decision_steps.reward, 'decision_steps.reward')
          reward = decision_steps.reward[i]
          # print(reward, 'reward')
          if reward <= -0.01:
            result = fitnesses[i]
          else:
            result = fitnesses[i] + reward
          # if i not in finish_order:
          #   if result >= 5000:
          #     fitnesses[i] = 5000 + (population - len(finish_order)) * 100
          #     finish_order.append(i)
          #   else:  
          fitnesses[i] = result
      if time_step > 500:  # all agents terminated
          done = True
          env.reset()
          if episode % 3 == 0 and episode != 0:
            print(fitnesses, 'fitnesses')
            print(np.mean(fitnesses), 'avg fitnesses')
            print(np.std(fitnesses), 'std fitnesses')
            best_ids = evaluate_fitness(fitnesses, 5)

            if save_weights:
              model.save_weights(f'model_weights-new-env/model-{episode}')
              
            weights = model.get_weights()
            new_weights = cross_over(weights, best_ids, population)
            model.set_weights(new_weights)
            fitnesses = np.full(population, 0)



  env.close()
  print("Closed environment")


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
