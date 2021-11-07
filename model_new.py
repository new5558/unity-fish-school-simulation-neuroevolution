from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from multiprocessing import Pool

def xavier_initialization(input: int, outputs: int, size):
    x = np.sqrt(6/(input + outputs)) 
    return np.random.uniform(-x, x, size)


def generate_layer(output_units: str, input_units: int, inputs_shape: int, outputs_shape: int):
	return {
		"weights": xavier_initialization(inputs_shape, outputs_shape, (input_units, output_units)),
		"bias": np.random.normal(0, 0.05, (1, output_units)),
		"activation": 'tanh'
		}


def predict_one_model(map_input):
	model = map_input['model']
	input_data = map_input['input_data']
	output = None
	for layer in model:
		# print(layer['weights'].shape, 'layer')
		# # print(input_data.shape, '??')
		# print(np.array(output).shape if output is not None else input_data[i].shape, 'input')
		# print(layer, input_data[i], 'layer')
		if output is None:
			z = np.dot(input_data, layer['weights']) + layer['bias']
			output = np.tanh(z)
		else:
			z = np.dot(output, layer['weights']) + layer['bias']
			output = np.tanh(z)
		# print(np.array(output).shape, 'output')
	return output.reshape((-1))

def predict(models, input_data):
	# result = []
	map_input = []
	for i in range(len(models)):
		map_input.append({"model": models[i], "input_data": input_data[i]})
	result = list(map(predict_one_model, map_input))
	return np.array(result)

def predict_concurrent(models, input_data):
	# result = []
	map_input = []
	for i in range(len(models)):
		map_input.append({"model": models[i], "input_data": input_data[i]})
	pool = Pool(2)
	print('preparation')
	result = pool.map(predict_one_model, map_input)
	print(result, 'result')
	# result = list(map(predict_one_model, map_input))
	return np.array(result)

# def predict(models, input_data):
# 	result = []
# 	for i in range(len(models)):
# 		output = None
# 		model = models[i]
# 		for layer in model:
# 			# print(layer['weights'].shape, 'layer')
# 			# # print(input_data.shape, '??')
# 			# print(np.array(output).shape if output is not None else input_data[i].shape, 'input')
# 			# print(layer, input_data[i], 'layer')
# 			if output is None:
# 				z = np.dot(input_data[i], layer['weights']) + layer['bias']
# 				output = np.tanh(z)
# 			else:
# 				z = np.dot(output, layer['weights']) + layer['bias']
# 				output = np.tanh(z)
# 			# print(np.array(output).shape, 'output')
# 		result.append(output.reshape((-1)))
# 	return np.array(result)

class NumpyModel:
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
		# if weights_path != None:
		# 	self.model.load_weights(weights_path)


	def generate_model(self):
		models = []
		for i in range(self.population):
			# input = keras.Input(shape=(self.inputs_shape,))
			# inputs.append(input)

			
			input_layer = generate_layer(8, self.inputs_shape, self.inputs_shape, self.outputs_shape)
			hidden_layer_1 = generate_layer(8, 8, self.inputs_shape, self.outputs_shape)
			hidden_layer_2 = generate_layer(4, 8, self.inputs_shape, self.outputs_shape)
			output_layer = generate_layer(self.outputs_shape, 4, self.inputs_shape, self.outputs_shape)

			model = [input_layer, hidden_layer_1, hidden_layer_2, output_layer]
			models.append(model)
		return models

	def __reshape_observation(self, observation):
		return np.array(observation).reshape(1, self.inputs_shape)


	def __transform_weights(self, weights_bias):
		new_weights_bias = []
		for model in weights_bias:
			model_weights = []
			model_bias = []
			for layer in model:
				model_weights.append(layer['weights'])
				model_bias.append(layer['bias'])
			new_weights_bias.append([model_weights, model_bias])
		return np.array(new_weights_bias)


	def generate_action(self, observation):
		inputs = list(map(self.__reshape_observation, list(observation))) 
		result = predict(self.model, inputs)
		# result = predict_concurrent(self.model, inputs)
		# action = np.array(result).reshape((self.population, self.outputs_shape))
		# print(result, result.shape, 'result')
		return result

	def get_weights(self, original = False):
	 	weights = self.model
	 	if original == True:
	 		return weights
	 	else:
	 		return self.__transform_weights(weights)

	def set_weights(self, weights, original = False):
		print(weights[0])
		if original == True:
			self.model = weights
		else:
			for i in range(len(weights)):
				weight, bias = weights[i]
				for j in range(len(weight)):
					self.model[i][j]['weights'] = weight[j]
					self.model[i][j]['bias'] = bias[j]
	 	

	def save_weights(self, path):
	 	#  return self.model.save_weights(path)
		return