from typing import Dict, Any
import numpy as np
import os
import json
from pathlib import Path


class Pooling:
	"""
	Pooling class for generating sentence embeddings from token embeddings using various pooling strategies.
	This is the numpy version adapted from the original Sentence Transformers' Pooling class.
	:param word_embedding_dimension: The dimensionality of the word embeddings.
	:param pooling_mode: The pooling mode to be used. Options include 'cls', 'lasttoken', 'max', 'mean',
		'mean_sqrt_len_tokens', and 'weightedmean'.
	:param pooling_mode_cls_token: Whether to include the CLS token in pooling. Defaults to False.
	:param pooling_mode_max_tokens: Whether to use max pooling. Defaults to False.
	:param pooling_mode_mean_tokens: Whether to use mean pooling. Defaults to True.
	:param pooling_mode_mean_sqrt_len_tokens: Whether to use mean pooling with square root of the number of tokens.
		Defaults to False.
	:param pooling_mode_weightedmean_tokens: Whether to use weighted mean pooling. Defaults to False.
	:param pooling_mode_lasttoken: Whether to use the last token. Defaults to False.
	:param include_prompt: Whether to include prompt tokens in pooling. Defaults to True.
	 Methods:
		 __repr__(): Returns a string representation of the Pooling object.
		 get_pooling_mode_str(): Returns the pooling mode as a string.
		 apply_pooling(features: Dict[str, np.ndarray]) -> np.ndarray:
			 Applies the configured pooling strategy to the input features and returns the resulting embeddings.
		 get_sentence_embedding_dimension() -> int:
			 Returns the dimensionality of the resulting sentence embeddings.
		 get_config_dict() -> Dict[str, Any]:
			 Returns a dictionary containing the configuration of the Pooling object.
		 save(output_path: str): Saves the configuration of the Pooling object to a JSON file.
		 load(input_path: str) -> Pooling: Loads a Pooling object from a JSON file.
	 Example:
		 # Initialize Pooling object
		 pooling = Pooling(word_embedding_dimension=768, pooling_mode='mean')
		 # Apply pooling to token embeddings
		 sentence_embeddings = pooling.apply_pooling(features={'token_embeddings': token_embeddings,
															   'attention_mask': attention_mask})
	"""
	POOLING_MODES = (
		"cls",
		"lasttoken",
		"max",
		"mean",
		"mean_sqrt_len_tokens",
		"weightedmean",
	)

	def __init__(
			self,
			word_embedding_dimension: int = None,
			pooling_mode: str = None,
			pooling_mode_cls_token: bool = False,
			pooling_mode_max_tokens: bool = False,
			pooling_mode_mean_tokens: bool = True,
			pooling_mode_mean_sqrt_len_tokens: bool = False,
			pooling_mode_weightedmean_tokens: bool = False,
			pooling_mode_lasttoken: bool = False,
			include_prompt=True,
	) -> None:
		self.config_keys = [
			"word_embedding_dimension",
			"pooling_mode_cls_token",
			"pooling_mode_mean_tokens",
			"pooling_mode_max_tokens",
			"pooling_mode_mean_sqrt_len_tokens",
			"pooling_mode_weightedmean_tokens",
			"pooling_mode_lasttoken",
			"include_prompt",
		]

		if pooling_mode is not None:  # Set pooling mode by string
			pooling_mode = pooling_mode.lower()

			if pooling_mode not in self.POOLING_MODES:
				raise ValueError(
					f"Set invalid pooling mode: {pooling_mode}. Valid pooling modes are: {self.POOLING_MODES}."
				)

			pooling_mode_cls_token = pooling_mode == "cls"
			pooling_mode_max_tokens = pooling_mode == "max"
			pooling_mode_mean_tokens = pooling_mode == "mean"
			pooling_mode_mean_sqrt_len_tokens = pooling_mode == "mean_sqrt_len_tokens"
			pooling_mode_weightedmean_tokens = pooling_mode == "weightedmean"
			pooling_mode_lasttoken = pooling_mode == "lasttoken"

		self.word_embedding_dimension = word_embedding_dimension
		self.pooling_mode_cls_token = pooling_mode_cls_token
		self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
		self.pooling_mode_max_tokens = pooling_mode_max_tokens
		self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
		self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
		self.pooling_mode_lasttoken = pooling_mode_lasttoken

		self.include_prompt = include_prompt

		pooling_mode_multiplier = sum(
			[
				pooling_mode_cls_token,
				pooling_mode_max_tokens,
				pooling_mode_mean_tokens,
				pooling_mode_mean_sqrt_len_tokens,
				pooling_mode_weightedmean_tokens,
				pooling_mode_lasttoken,
			]
		)
		
		if isinstance(word_embedding_dimension, int):
			self.pooling_output_dimension = pooling_mode_multiplier * word_embedding_dimension
		else:
			self.pooling_output_dimension = None

	def __repr__(self):
		return "Pooling({})".format(self.get_config_dict())

	def get_pooling_mode_str(self) -> str:
		"""
		Returns the pooling mode as string
		"""
		modes = []
		if self.pooling_mode_cls_token:
			modes.append("cls")
		if self.pooling_mode_mean_tokens:
			modes.append("mean")
		if self.pooling_mode_max_tokens:
			modes.append("max")
		if self.pooling_mode_mean_sqrt_len_tokens:
			modes.append("mean_sqrt_len_tokens")
		if self.pooling_mode_weightedmean_tokens:
			modes.append("weightedmean")
		if self.pooling_mode_lasttoken:
			modes.append("lasttoken")

		return "+".join(modes)

	def __call__(
			self,
			features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
		"""
		Applies the configured pooling strategy to the input features and returns the resulting embeddings.
		:param features: A dictionary containing the token embeddings and attention masks.
		:type features: Dict[str, np.ndarray]
		:return: The sentence embeddings generated by the pooling operation.
		:rtype: np.ndarray
		"""
		token_embeddings = features["token_embeddings"]
		attention_mask = features["attention_mask"]

		if not self.include_prompt and "prompt_length" in features:
			attention_mask[:, : features["prompt_length"]] = 0

		# Pooling strategy
		output_vectors = []
		if self.pooling_mode_cls_token:
			cls_token = features.get(
				"cls_token_embeddings", token_embeddings[:, 0]
			)  # Take first token by default
			output_vectors.append(cls_token)
		if self.pooling_mode_max_tokens:
			# Set padding tokens to large negative value
			token_embeddings[attention_mask == 0] = -1e9
			max_over_time = np.max(token_embeddings, axis=1)
			output_vectors.append(max_over_time)
		if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
			sum_embeddings = np.sum(token_embeddings * attention_mask[:, :, np.newaxis], axis=1)

			# If tokens are weighted, feature 'token_weights_sum' will be present
			if "token_weights_sum" in features:
				sum_mask = features["token_weights_sum"].reshape(-1, 1)
			else:
				sum_mask = np.sum(attention_mask, axis=1, keepdims=True)

			sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)

			if self.pooling_mode_mean_tokens:
				output_vectors.append(sum_embeddings / sum_mask)
			if self.pooling_mode_mean_sqrt_len_tokens:
				output_vectors.append(sum_embeddings / np.sqrt(sum_mask))
		# Add implementations for other pooling modes here

		output_vector = np.concatenate(output_vectors, axis=1)
		features["sentence_embedding"] = output_vector
		return features

	def get_sentence_embedding_dimension(self) -> int:
		"""
		Returns the dimensionality of the resulting sentence embeddings.
		:return: The dimensionality of the sentence embeddings.
		:rtype: int
		"""
		return self.pooling_output_dimension

	def get_config_dict(self) -> Dict[str, Any]:
		"""
		Returns a dictionary containing the configuration of the Pooling object.
		:return: The configuration dictionary.
		:rtype: Dict[str, Any]
		"""
		return {key: self.__dict__[key] for key in self.config_keys}

	def save(self, output_path):
		"""
		Saves the configuration of the Pooling object to a JSON file.
		:param output_path: The path where the configuration JSON file will be saved.
		:type output_path: str
		"""
		with open(os.path.join(output_path, "config.json"), "w") as fOut:
			json.dump(self.get_config_dict(), fOut, indent=2)

	@staticmethod
	def load(input_path: str or Path):
		"""
		Loads a Pooling object from a JSON file.
		:param input_path: The path to the directory containing the configuration JSON file.
		:type input_path: str
		:return: The loaded Pooling object.
		:rtype: Pooling
		"""
		config_file_path = Path(input_path, "config.json")
		with open(str(config_file_path)) as fIn:
			config = json.load(fIn)

		return Pooling(**config)