from typing import Optional, Union, List, Literal
from pathlib import Path
import numpy as np
import json
from light_embed.utils.model import get_onnx_model_dir
from light_embed.utils.functions import normalize, quantize_embeddings
from light_embed.modules import OnnxText
from light_embed.modules import FastTokenizer
import logging

logger = logging.getLogger(__name__)

class TextEmbedding:
	"""
	TextEmbedding class for generating embeddings from text using Hugging Face models.

	:param model_name_or_path: The name or path of the pre-trained Hugging Face model.
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
	 	embedding = TextEmbedding(model_name_or_path='bert-base-uncased')
	 	embeddings = embedding.encode(sentences=['Hello world!', 'How are you?'])
	"""
	
	def __init__(
			self,
			model_name_or_path: str,
			cache_folder: Optional[str or Path] = None,
			quantize: bool = False,
			device: str = "cpu"
	) -> None:
		self.model_name_or_path = model_name_or_path
		self.session = None
		self.device = device

		self.model_dir = get_onnx_model_dir(
			model_name_or_path=model_name_or_path,
			cache_dir=cache_folder,
			quantize=quantize
		)
		
		# Load model description
		self._model_description = self._load_model_description()
		
		# Load sentence-transformers' onnx model
		self.model = self._load_onnx_model()
		model_input_names = self.model.model_input_names

		# Load tokenizer from file
		self.tokenizer = FastTokenizer.load(
			input_path=self.model_dir, model_input_names=model_input_names)
		
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
		
	def _load_onnx_model(self):
		"""
		Load the ONNX model specified in the model description.

		:return: OnnxText: An instance of the loaded ONNX model.
		"""
		onnx_model_file = self._model_description.get(
			"model_file", "model.onnx")
		onnx_model_path = Path(self.model_dir, onnx_model_file)
		
		# Load sentence-transformers' onnx model
		model = OnnxText.load(input_path=onnx_model_path, device=self.device)
		return model
		
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

			onnx_result = self.model.apply(features)
			
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
	
	def get_embedding_dimension(self):
		"""
		Retrieve the embedding dimension from the model description.

		This method fetches the value associated with the "embedding_dim" key from
		the model description if it is stored as a dictionary. The embedding dimension
		is an important parameter in various machine learning models, particularly those
		involving embeddings such as NLP models or recommendation systems.

		Returns:
			int or None: The embedding dimension if present in the model description
			dictionary; otherwise, None.
		"""
		if isinstance(self._model_description, dict):
			return self._model_description.get("embedding_dim")
		else:
			return None
	
	def get_max_seq_length(self):
		"""
		Retrieve the maximum sequence length from the model description.

		This method fetches the value associated with the "max_seq_length" key from
		the model description if it is stored as a dictionary. The maximum sequence
		length is an important parameter in various machine learning models, especially
		those involving sequential data such as NLP models or time series analysis.

		Returns:
			int or None: The maximum sequence length if present in the model description
			dictionary; otherwise, None.
		"""
		if isinstance(self._model_description, dict):
			return self._model_description.get("max_seq_length")
		else:
			return None