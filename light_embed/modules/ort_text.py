from typing import Dict, Union
import numpy as np
from pathlib import Path
from .ort_base import OrtModel
from light_embed.utils import REPO_ORG_NAME
import logging

logger = logging.getLogger(__name__)

managed_models = [
	# sentence-transformers models
	{
		"model_name": f"{REPO_ORG_NAME}/all-mpnet-base-v2-onnx",
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
		"model_name": f"{REPO_ORG_NAME}/all-mpnet-base-v1-onnx",
		"base_model": "sentence-transformers/all-mpnet-base-v1",
		"quantized": "false",
		"modules": [
			{
				"type": "onnx_model",
				"path": "model.onnx"
			}
		]
	},
	{
		"model_name": f"{REPO_ORG_NAME}/LaBSE-onnx",
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
		"model_name": f"{REPO_ORG_NAME}/sentence-t5-base-onnx",
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
		"model_name": f"{REPO_ORG_NAME}/all-MiniLM-L12-v2-onnx",
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
		"model_name": f"{REPO_ORG_NAME}/sentence-bert-swedish-cased-onnx",
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
		"model_name": "BAAI/bge-base-en",
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
		"model_name": "BAAI/bge-base-en-v1.5",
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
		"model_name": "BAAI/bge-small-en-v1.5",
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


class OrtText(OrtModel):
	OUTPUT_NAMES = ("token_embeddings", "sentence_embedding")
	
	output_name_map = {
		"last_hidden_state": "token_embeddings"
	}

	def __init__(
		self,
		model_path: Union[str, Path],
		**kwargs
	):
		super().__init__(model_path, **kwargs)
		
		output_name_map = kwargs.get("output_name_map")
		if output_name_map is not None and isinstance(output_name_map, dict):
			self.output_name_map.update(output_name_map)
	
	def __call__(
		self,
		features: Dict[str, np.array],
		**kwargs
	) -> Dict[str, np.ndarray]:
		ort_output = super().__call__(features)
		
		out_features = dict()
		for i, output_name in enumerate(self.model_output_names):
			mapped_name = self.output_name_map.get(output_name, output_name)
			if mapped_name in self.OUTPUT_NAMES:
				out_features[mapped_name] = ort_output[i]
		
		out_features["attention_mask"] = features["attention_mask"]
		
		return out_features
