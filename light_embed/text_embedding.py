from typing import Optional, Union, List, Literal
from pathlib import Path
import numpy as np
import json
from light_embed.utils.model import get_managed_model_config, download_onnx_model
from light_embed.utils.functions import normalize, quantize_embeddings
from light_embed.modules import OrtText, managed_text_models, Pooling, Normalize
from light_embed.modules import Tokenizer
import logging

logger = logging.getLogger(__name__)

class TextEmbedding:
	"""
	TextEmbedding class for generating embeddings from text using Hugging Face models.

	:param model_name: The name of the pre-trained Hugging Face model.
	:param cache_folder: Optional. Folder to cache the downloaded model files. Defaults to None.
	:param quantize: Optional. Whether to quantize the ONNX model for performance. Defaults to False.
	:param device: Optional. Device to run inference on, e.g., 'cpu' or 'cuda'. Defaults to 'cpu'.

	Attributes:
		session: ONNX runtime session for running inference.
		device: Device for running inference.
		tokenizer: Tokenizer for the Hugging Face model.
		pooling_model: Pooling model for aggregating token embeddings.

	Methods:
	 	encode(sentences, batch_size=32, normalize_output=True):
		 	Encodes input sentences into embeddings.

	Example:
	 	embedding = TextEmbedding(model_name='bert-base-uncased')
	 	embeddings = embedding.encode(sentences=['Hello world!', 'How are you?'])
	"""
	
	def __init__(
			self,
			model_name: str,
			cache_folder: Optional[str or Path] = None,
			quantize: Union[bool, str] = None,
			device: Optional[str] = None,
			**kwargs
	) -> None:
		self.model_name = model_name
		self.session = None
		self.device = device
		self._transformer_config = {}
		
		onnx_file = kwargs.get("onnx_file")
		
		# if model_file is provided then use model config
		# given from command line
		# otherwise, use light-embed managed model config
		if onnx_file is not None:
			model_config = {
				"model_name": model_name,
				"onnx_file": onnx_file
			}
			
			pooling_config_path = kwargs.get("pooling_config_path")
			if isinstance(pooling_config_path, str):
				model_config["pooling_config_path"] = pooling_config_path
			else:
				pooling_mode = kwargs.get("pooling_mode")
				if isinstance(pooling_mode, str):
					model_config["pooling_mode"] = pooling_mode
			
			normalize_bool = kwargs.get("normalize", False)
			if isinstance(normalize_bool, bool):
				model_config["normalize"] = normalize_bool
		
		else:
			model_config = get_managed_model_config(
				base_model_name=model_name,
				quantize=quantize,
				managed_models=managed_text_models
			)
			
			if model_config is None:
				raise ValueError(
					f"model {model_name} with quantize={quantize} is not supported."
				)

		self.model_config = model_config

		self.model_dir = download_onnx_model(
			model_config=model_config,
			cache_dir=cache_folder
		)
		
		# Load sentence-transformers' onnx model
		self.modules = self._load_model()
		model_input_names = self.modules[0].model_input_names

		# Load tokenizer from file
		self.tokenizer = Tokenizer.load(
			input_path=self.model_dir, model_input_names=model_input_names)
		
		model_max_length = self.tokenizer.model_max_length
		max_position_embeddings = self._transformer_config.get(
			"max_position_embeddings")
		if max_position_embeddings is not None and model_max_length is not None:
			max_seq_length = min(
				max_position_embeddings,
				model_max_length
			)
		else:
			max_seq_length = None
		self.max_seq_length = max_seq_length
		
	def _load_model_description(self):
		"""
		Load the model description from a JSON file.

		:return: dict or None: The model description as a dictionary if the file exists and is
        successfully parsed, otherwise None.
		"""
		model_description_json_path = Path(
			self.model_dir, "model_description.json")
		if Path(model_description_json_path).exists():
			with open(model_description_json_path) as fIn:
				model_description = json.load(fIn)
		else:
			model_description = None
		return model_description
	
	def _create_modules_config(self):
		modules_config = []
		onnx_config = {
			"type": "onnx_model",
			"path": self.model_config.get("onnx_file"),
			"output_name_map": self.model_config.get("onnx_output_map")
		}
		modules_config.append(onnx_config)
		
		pooling_config_path = self.model_config.get("pooling_config_path")
		pooling_mode = self.model_config.get("pooling_mode")
		if isinstance(pooling_config_path, str):
			pooling_config = {
				"type": "pooling",
				"pooling_config_path": pooling_config_path
			}
		elif isinstance(pooling_mode, str):
			pooling_config = {
				"type": "pooling",
				"pooling_mode": pooling_mode
			}
		else:
			pooling_config = None
		
		if pooling_config is not None:
			modules_config.append(pooling_config)
			
		normalize_bool = self.model_config.get("normalize", True)
		if normalize_bool:
			modules_config.append({"type": "normalize"})
			
		return modules_config
	
	def _load_model(self):
		config_json_path = Path(self.model_dir, "config.json")
		if config_json_path.exists():
			with open(str(config_json_path)) as fIn:
				self._transformer_config = json.load(fIn)

		modules_config = self._create_modules_config()
		
		modules = []
		for module_config in modules_config:
			module_type = module_config.get("type").lower()
			if module_type == "onnx_model":
				module_path = Path(self.model_dir, module_config.get("path"))
				output_name_map = module_config.get("output_name_map")
				module = OrtText.load(
					input_path=module_path, output_name_map=output_name_map,
					device=self.device)
			elif module_type == "pooling":
				pooling_config_path = self.model_config.get("pooling_config_path")
				pooling_mode = self.model_config.get("pooling_mode")
				
				if isinstance(pooling_config_path, str):
					module_path = Path(self.model_dir, pooling_config_path)
					module = Pooling.load(input_path=module_path)
				elif isinstance(pooling_mode, str):
					module = Pooling(pooling_mode=pooling_mode)
			elif module_type == "normalize":
				module = Normalize()
			else:
				continue

			modules.append(module)
		
		return modules
		
	def tokenize(
		self,
		texts: List[str]
	):
		"""
		Tokenize a list of texts using the model's tokenizer.

		:param: texts (List[str]): A list of strings to be tokenized.

		:return: List: A list of tokenized representations of the input texts.
		"""
		return self.tokenizer.tokenize(texts)

	def apply(self, features):
		for module in self.modules:
			features = module(features)
		
		return features
	
	def encode(
		self,
		sentences: Union[str, List[str]],
		batch_size: int = 32,
		output_value: Optional[Literal["sentence_embedding", "token_embeddings"]] = "sentence_embedding",
		precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
		return_as_array: bool = True,
		return_as_list: bool = False,
		normalize_embeddings: bool = False
	) -> np.ndarray:
		"""
		Encodes input sentences into embeddings.

		:param return_as_array:
		:param return_as_list:
		:param precision:
		:param output_value:
		:param sentences: Input sentences to be encoded, either a single string or a list of strings.
		:param batch_size: Batch size for encoding. Defaults to 32.
		:param normalize_embeddings: Whether to normalize output embeddings. Defaults to True.

		:return: Encoded embeddings as a numpy array.
		"""
		input_was_string = False
		if isinstance(sentences, str) or not hasattr(
			sentences, "__len__"
		):  # Cast an individual sentence to a list with length 1
			sentences = [sentences]
			input_was_string = True
		
		all_embeddings = []
		length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
		sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
		
		for start_index in range(0, len(sentences), batch_size):
			sentences_batch = sentences_sorted[start_index: start_index + batch_size]
			features = self.tokenize(sentences_batch)

			onnx_result = self.apply(features)
			
			if output_value == "token_embeddings":
				embeddings = onnx_result.get("token_embeddings")
			elif output_value is None:
				embeddings = []
				for sent_idx in range(len(onnx_result.get("sentence_embedding"))):
					row = {name: onnx_result[name][sent_idx] for name in onnx_result}
					embeddings.append(row)
			else:  # Sentence embeddings
				embeddings = onnx_result.get(output_value)
			
				if normalize_embeddings:
					embeddings = normalize(embeddings)
			
			all_embeddings.extend(embeddings)
		
		all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
		
		if precision and precision != "float32":
			all_embeddings = quantize_embeddings(all_embeddings, precision=precision)
		
		if return_as_array:
			all_embeddings = np.asarray(all_embeddings)
		elif return_as_list:
			all_embeddings = list(all_embeddings)
		
		if input_was_string:
			all_embeddings = all_embeddings[0]
		
		return all_embeddings
	
	@staticmethod
	def _text_length(
		text: Union[List[int], List[List[int]]]):
		"""
		Help function to get the length for the input text. Text can be either
		a list of ints (which means a single text as input), or a tuple of list of ints
		(representing several text inputs to the model).
		"""
		
		if isinstance(text, dict):  # {key: value} case
			return len(next(iter(text.values())))
		elif not hasattr(text, "__len__"):  # Object has no len() method
			return 1
		elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
			return len(text)
		else:
			return sum([len(t) for t in text])  # Sum of length of individual strings