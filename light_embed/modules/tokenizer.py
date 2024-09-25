from typing import Union, List, Dict
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
		
	def __call__(self, sentences: Union[str, List[str]]):
		return self.tokenize(sentences=sentences)
	
	def tokenize(
		self, sentences: Union[str, List[str]]) -> Dict[str, np.ndarray]:
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
		
		features = {
			"input_ids": np.array(input_ids, dtype=np.int64)
		}
		if "attention_mask" in self.model_input_names:
			attention_mask = np.array([e.attention_mask for e in encoded])
			features["attention_mask"] = np.array(attention_mask, dtype=np.int64)
		
		if "token_type_ids" in self.model_input_names:
			token_type_ids = np.array([e.type_ids for e in encoded])
			features["token_type_ids"] = np.array(token_type_ids, dtype=np.int64)

		return features
	
	@property
	def model_max_length(self):
		return self.tokenizer.truncation.get("max_length")
	
	@staticmethod
	def load(
		input_path: Union[str, Path],
		max_length: int = None, **kwargs) -> tokenizers.Tokenizer:
		
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
		
		required_files = [
			"config.json", "tokenizer.json",
			"tokenizer_config.json", "special_tokens_map.json"
		]
		
		missing_files = [
			file for file in required_files
			if not Path(input_path, file).exists()
		]

		if missing_files:
			raise ValueError(
				f"Could not find the following files in {input_path}: "
				f"{', '.join(missing_files)}"
			)
		
		# Check and load required files
		config_data = dict()
		for file_name in required_files:
			if file_name != "tokenizer.json":
				file_path = str(Path(input_path, file_name))
				key = file_name.rsplit('.', 1)[0]
				with open(file_path) as f:
					config_data[key] = json.load(f)
		
		# Load tokenizer from file
		tokenizer_path = str(Path(input_path, "tokenizer.json"))
		tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
		
		# Enable truncation and padding
		model_max_length = config_data["tokenizer_config"].get("model_max_length")
		if isinstance(max_length, int) and model_max_length is not None:
			model_max_length = min(model_max_length, max_length)
		elif isinstance(max_length, int):
			model_max_length = max_length
		
		if model_max_length is not None:
			tokenizer.enable_truncation(max_length=model_max_length)

		pad_token_id = config_data["config"].get("pad_token_id", 0)
		pad_token = config_data["tokenizer_config"]["pad_token"]
		tokenizer.enable_padding(
			pad_id=pad_token_id, pad_token=pad_token
		)
		
		# Add special tokens
		for token in config_data["special_tokens_map"].values():
			if isinstance(token, str):
				tokenizer.add_special_tokens([token])
			elif isinstance(token, dict):
				tokenizer.add_special_tokens(
					[tokenizers.AddedToken(**token)])
		
		return Tokenizer(tokenizer, **kwargs)
