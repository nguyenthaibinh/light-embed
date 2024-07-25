from typing import Dict, Union
import numpy as np
from pathlib import Path
from .onnx_base import OnnxModel
import logging

logger = logging.getLogger(__name__)

supported_text_embedding_models = [
	{
		"model_name": "sentence-transformers/all-MiniLM-L6-v2",
		"quantized": "false",
		"ort_output_keys": ["token_embeddings", "sentence_embedding"],
		"modules": [
			{
				"type": "onnx_model",
				"path": "onnx/model.onnx"
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
	ort_output_keys = ["token_embeddings"]

	def __init__(
		self,
		model_path: Union[str, Path],
		**kwargs
	):
		super().__init__(model_path, **kwargs)

		ort_output_keys = kwargs.get("ort_output_keys", None)
		if isinstance(ort_output_keys, list):
			self.ort_output_keys = ort_output_keys
	
	def apply(
		self,
		features: Dict[str, np.array],
		**kwargs
	):
		ort_output = super().apply(features)
		
		out_features = {
			key: ort_output[i] for i, key in enumerate(self.ort_output_keys)
		}
		
		if "attention_mask" not in out_features:
			out_features["attention_mask"] = features["attention_mask"]
		
		return out_features
	
	@staticmethod
	def load(input_path: Union[str, Path], **kwargs):
		return OnnxText(input_path, **kwargs)
