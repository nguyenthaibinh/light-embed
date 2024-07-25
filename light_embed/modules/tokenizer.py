from typing import Union, List
from pathlib import Path
import tokenizers
import json
import numpy as np


class Tokenizer:
	"""
	Initialize the model with a tokenizer and optional configuration.

	Parameters:
		tokenizer (tokenizers.Tokenizer): The tokenizer to be used by the model.
		**kwargs: Additional keyword arguments for configuration.
			- model_input_names (list, optional): A list of input names expected by the model.
			  Defaults to `self.model_input_names`.
	"""
	model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
	
	def __init__(
		self,
		tokenizer: tokenizers.Tokenizer,
		**kwargs
	):
		self.tokenizer = tokenizer
		self.model_input_names = kwargs.pop(
			"model_input_names", self.model_input_names
		)
	
	def tokenize(self, sentences: Union[str, List[str]]):
		"""
		Tokenize input sentences using the model's tokenizer.

		Parameters:
			sentences (Union[str, List[str]]): A single sentence (str) or a list
				of sentences (List[str]) to be tokenized.

		Returns:
			dict: A dictionary with the following keys:
				- "input_ids" (np.ndarray): Array of input IDs.
				- "attention_mask" (np.ndarray, optional): Array of attention masks,
					if "attention_mask" is in `self.model_input_names`.
				- "token_type_ids" (np.ndarray, optional): Array of token type IDs,
					if "token_type_ids" is in `self.model_input_names`.
		"""
		
		if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
			# Cast an individual sentence to a list with length 1
			sentences = [sentences]
		
		encoded = self.tokenizer.encode_batch(sentences)
		input_ids = np.array([e.ids for e in encoded])
		attention_mask = np.array([e.attention_mask for e in encoded])
		
		features = {
			"input_ids": np.array(input_ids, dtype=np.int64)
		}
		if "attention_mask" in self.model_input_names:
			features["attention_mask"] = np.array(attention_mask, dtype=np.int64)
		
		if "token_type_ids" in self.model_input_names:
			token_type_ids = np.array([e.type_ids for e in encoded])
			features["token_type_ids"] = np.array(token_type_ids, dtype=np.int64)

		return features
	
	@staticmethod
	def load(
		input_path: Union[str, Path],
		max_length: int = 512, **kwargs) -> tokenizers.Tokenizer:
		
		"""
		Load a tokenizer from the specified input path.

		Parameters:
			input_path (Union[str, Path]): The directory containing the tokenizer configuration files.
			max_length (int, optional): The maximum sequence length for truncation. Defaults to 512.
			**kwargs: Additional keyword arguments passed to the Tokenizer constructor.

		Returns:
			tokenizers.Tokenizer: An initialized tokenizer ready for use.

		Raises:
			ValueError: If any of the required configuration files (config.json, tokenizer.json,
			tokenizer_config.json, special_tokens_map.json) are missing from the input path.
		"""
		
		config_path = Path(input_path, "config.json")
		if not config_path.exists():
			raise ValueError(f"Could not find config.json in {input_path}")
		
		tokenizer_path = Path(input_path, "tokenizer.json")
		if not tokenizer_path.exists():
			raise ValueError(f"Could not find tokenizer.json in {input_path}")
		
		tokenizer_config_path = Path(input_path, "tokenizer_config.json")
		if not tokenizer_config_path.exists():
			raise ValueError(f"Could not find tokenizer_config.json in {input_path}")
		
		tokens_map_path = Path(input_path, "special_tokens_map.json")
		if not tokens_map_path.exists():
			raise ValueError(f"Could not find special_tokens_map.json in {input_path}")
		
		with open(str(config_path)) as config_file:
			config = json.load(config_file)
		
		with open(str(tokenizer_config_path)) as tokenizer_config_file:
			tokenizer_config = json.load(tokenizer_config_file)
		
		with open(str(tokens_map_path)) as tokens_map_file:
			tokens_map = json.load(tokens_map_file)
		
		tokenizer_path_str = str(tokenizer_path)
		tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path_str)
		tokenizer.enable_truncation(
			max_length=min(tokenizer_config["model_max_length"], max_length)
		)
		tokenizer.enable_padding(
			pad_id=config.get("pad_token_id", 0), pad_token=tokenizer_config["pad_token"]
		)
		
		for token in tokens_map.values():
			if isinstance(token, str):
				tokenizer.add_special_tokens([token])
			elif isinstance(token, dict):
				tokenizer.add_special_tokens(
					[tokenizers.AddedToken(**token)])
		
		return Tokenizer(tokenizer, **kwargs)
