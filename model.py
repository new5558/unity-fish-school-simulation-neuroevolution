from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

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


	def generate_model(self):
		outputs = []
		inputs = []
		for i in range(self.population):
			input = keras.Input(shape=(self.inputs_shape,))
			inputs.append(input)
			x = layers.Dense(12, activation="tanh", bias_initializer='random_normal')(input)
			# x = layers.Dense(8, activation="tanh", bias_initializer='random_normal')(x)
			# x = layers.Dense(4, activation="tanh", bias_initializer='random_normal')(x)
			outputs.append(layers.Dense(self.outputs_shape, activation="tanh", bias_initializer='random_normal')(x))
			# concatted = tf.keras.layers.Concatenate()(outputs)
		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(
		loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		optimizer=keras.optimizers.RMSprop(), # We don't actually use optimizer, loss, and accuracy
		metrics=["accuracy"],
		)
		return model

	def __reshape_observation(self, observation):
		return np.array(observation).reshape(1, self.inputs_shape)


	def __transform_weights(self, weights_bias):
		networks = np.array(weights_bias).reshape(-1, self.population, 2)
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
		result = self.model.predict(inputs)
		action = np.array(result).reshape((self.population, self.outputs_shape))
		return action

	def get_weights(self, original = False):
		weights = self.model.get_weights()
		print(self.__transform_weights(weights)[0],  'weights')
		if original == True:
			return weights
		else:
			return self.__transform_weights(weights)

	def set_weights(self, weights, original = False):
	 	if original == True:
	 		return self.model.set_weights(weights)
	 	else:
	 		weights_keras = self.__detransform_weights(weights)
	 		return self.model.set_weights(weights_keras)
	 	

	def save_weights(self, path):
	 	 return self.model.save_weights(path)



