from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import os
import tempfile

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def create_dict(observation):
	result = {}
	for i in range(len(observation)):
		result[f"input_{i + 1}"] = tf.constant(observation[i])
	return result

def get_from_dict(dict_action):
	result = []
	for i in range(len(dict_action.keys())):
		result.append(dict_action[f"output_{i + 1}"].numpy()) 
	return np.array(result)

class KerasModel:
	"""Model Class"""

	model = None
	population = None
	inputs_shape = None
	outputs_shape = None

	def __init__(self, population, inputs_shape, outputs_shape, weights_path = None):
		self.population = population
		self.outputs_shape = outputs_shape
		self.inputs_shape = inputs_shape
		self.model = self.generate_model()
		if weights_path != None:
			self.model.load_weights(weights_path)
		self.tensor_model = self.model_to_tensor(self.model)


	def generate_model(self):		
		inputs = []
		outputs = []
		for i in range(self.population):
			input = keras.Input(shape=(self.inputs_shape,), name=f"input_{i + 1}")
			inputs.append(input)
			x = layers.Dense(16, activation="tanh", bias_initializer='random_normal')(input)
			x = layers.Dense(16, activation="tanh", bias_initializer='random_normal')(input)
			x = layers.Dense(8, activation="relu", bias_initializer='random_normal')(input)
			output = layers.Dense(self.outputs_shape, activation="tanh", bias_initializer='random_normal', name=f"output_{i + 1}")(x)
			outputs.append(output)
			# concatted = tf.keras.layers.Concatenate()(outputs)
		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		optimizer=keras.optimizers.RMSprop(), # We don't actually use optimizer, loss, and accuracy
		metrics=["accuracy"],
		)
		return model

	def model_to_tensor(self, model):
		tmpdir = tempfile.mkdtemp()
		save_path = os.path.join(tmpdir, f"norapat_net/1/")
		tf.saved_model.save(model, save_path)

		loaded = tf.saved_model.load(save_path)
		return loaded


	def __reshape_observation(self, observation):
		return np.array(observation).reshape(1, self.inputs_shape)


	def __transform_weights(self, weights_bias):
		networks = np.array(weights_bias, dtype=object).reshape(-1, self.population, 2)
		networks = np.transpose(networks, (1, 0, 2))
		networks = np.transpose(networks, (0, 2, 1))
		return networks

	def __detransform_weights(self, networks):
		networks = np.transpose(networks, (0, 2, 1))
		networks = np.transpose(networks, (1, 0, 2))
		layers = networks.reshape(-1)
		return layers

	def generate_action(self, observation):
		inputs = list(map(self.__reshape_observation, list(observation))) 
		feed_dict = create_dict(inputs)
		result = self.tensor_model.signatures["serving_default"](**feed_dict)
		action = get_from_dict(result).reshape((self.population, self.outputs_shape))
		return action

	def get_weights(self, original = False):
		weights = self.model.get_weights()
		if original == True:
			return weights
		else:
			return self.__transform_weights(weights)

	def set_weights(self, weights, original = False):
		set_weights_result = None
		if original == True:
	 		set_weights_result = self.model.set_weights(weights)
		else:
			weights_keras = self.__detransform_weights(weights)
			set_weights_result = self.model.set_weights(weights_keras)

		self.tensor_model = self.model_to_tensor(self.model)
		return set_weights_result
	 	

	def save_weights(self, path):
	 	 return self.model.save_weights(path)



