from typing import Dict, Union
import numpy as np
from pathlib import Path
from .onnx_base import OnnxModel
import logging

logger = logging.getLogger(__name__)

supported_text_embedding_models = [
	# sentence-transformers models
	{
		"model_name": "LightEmbed/all-mpnet-base-v2-onnx",
		"base_model": "sentence-transformers/all-mpnet-base-v2",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "model.onnx"
			}
		]
	},
	{
		"model_name": "LightEmbed/all-mpnet-base-v1-onnx",
		"base_model": "sentence-transformers/all-mpnet-base-v1",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "model.onnx"
			},
			{
				"type": "pooling",
				"path": "1_Pooling"
			},
			{
				"type": "normalize",
				"path": "2_Normalize"
			}
		]
	},
	{
		"model_name": "LightEmbed/LaBSE-onnx",
		"base_model": "sentence-transformers/LaBSE",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "model.onnx"
			},
			{
				"type": "pooling",
				"path": "1_Pooling"
			},
			{
				"type": "normalize",
				"path": "2_Normalize"
			}
		]
	},
	{
		"model_name": "LightEmbed/sentence-t5-base-onnx",
		"base_model": "sentence-transformers/sentence-t5-base",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "model.onnx"
			}
		]
	},
	{
		"model_name": "sentence-transformers/all-MiniLM-L6-v2",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "onnx/model.onnx"
			}
		]
	},
	{
		"model_name": "LightEmbed/all-MiniLM-L12-v2-onnx",
		"base_model": "sentence-transformers/all-MiniLM-L12-v2",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "model.onnx"
			}
		]
	},
	{
		"model_name": "BAAI/bge-large-en-v1.5",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "onnx/model.onnx"
			},
			{
				"type": "pooling",
				"path": "1_Pooling"
			},
			{
				"type": "normalize",
				"path": "2_Normalize"
			}
		]
	},
	{
		"model_name": "snowflake/snowflake-arctic-embed-xs",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "onnx/model.onnx"
			},
			{
				"type": "pooling",
				"path": "1_Pooling"
			},
			{
				"type": "normalize",
				"path": "2_Normalize"
			}
		]
	},
	{
		"model_name": "LightEmbed/sentence-bert-swedish-cased-onnx",
		"base_model": "KBLab/sentence-bert-swedish-cased",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "model.onnx"
			}
		]
	},
	{
		"model_name": "jinaai/jina-embeddings-v2-base-en",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "model.onnx"
			},
			{
				"type": "pooling",
				"path": "1_Pooling"
			}
		]
	}
]


class OnnxText(OnnxModel):
	OUTPUT_NAMES = (
		"last_hidden_state",
		"token_embeddings",
		"sentence_embedding"
	)

	def __init__(
		self,
		model_path: Union[str, Path],
		**kwargs
	):
		super().__init__(model_path, **kwargs)
		
		if not any(element in self.model_output_names for element in self.OUTPUT_NAMES):
			raise ValueError(f"Onnx output features must contain one of these features {self.OUTPUT_NAMES}.")
	
	def apply(
		self,
		features: Dict[str, np.array],
		**kwargs
	):
		ort_output = super().apply(features)
		
		out_features = {
			key: ort_output[i] for i, key in enumerate(self.model_output_names)
		}
		
		if "token_embeddings" not in out_features and "last_hidden_state" in out_features:
			out_features["token_embeddings"] = out_features.pop('last_hidden_state')
		
		out_features["attention_mask"] = features["attention_mask"]
		
		return out_features
	
	@staticmethod
	def load(input_path: Union[str, Path], **kwargs):
		return OnnxText(input_path, **kwargs)
