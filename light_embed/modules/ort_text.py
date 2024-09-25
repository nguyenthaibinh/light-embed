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
		"onnx_file": "model.onnx",
	},
	{
		"model_name": f"{REPO_ORG_NAME}/all-mpnet-base-v1-onnx",
		"base_model": "sentence-transformers/all-mpnet-base-v1",
		"onnx_file": "model.onnx",
	},
	{
		"model_name": f"{REPO_ORG_NAME}/paraphrase-mpnet-base-v2-onnx",
		"base_model": "sentence-transformers/paraphrase-mpnet-base-v2",
		"onnx_file": "model.onnx",
	},
	{
		"model_name": f"{REPO_ORG_NAME}/LaBSE-onnx",
		"base_model": "sentence-transformers/LaBSE",
		"onnx_file": "model.onnx",
	},
	{
		"model_name": f"{REPO_ORG_NAME}/sentence-t5-base-onnx",
		"base_model": "sentence-transformers/sentence-t5-base",
		"onnx_file": "model.onnx"
	},
	{
		"model_name": f"{REPO_ORG_NAME}/all-MiniLM-L6-v2-onnx",
		"base_model": "sentence-transformers/all-MiniLM-L6-v2",
		"onnx_file": "model.onnx"
	},
	{
		"model_name": f"{REPO_ORG_NAME}/all-MiniLM-L12-v2-onnx",
		"base_model": "sentence-transformers/all-MiniLM-L12-v2",
		"onnx_file": "model.onnx"
	},
	# BAAI models
	{
		"model_name": "BAAI/bge-large-en-v1.5",
		"onnx_file": "onnx/model.onnx",
		"pooling_config_path": "1_Pooling",
		"normalize": True
	},
	{
		"model_name": "snowflake/snowflake-arctic-embed-xs",
		"onnx_file": "onnx/model.onnx",
		"pooling_config_path": "1_Pooling",
		"normalize": True
	},
	{
		"model_name": f"{REPO_ORG_NAME}/sentence-bert-swedish-cased-onnx",
		"base_model": "KBLab/sentence-bert-swedish-cased",
		"onnx_file": "model.onnx"
	},
	# BAAI models
	{
		"model_name": "BAAI/bge-base-en",
		"onnx_file": "onnx/model.onnx",
		"pooling_config_path": "1_Pooling",
		"normalize": True
	},
	{
		"model_name": "BAAI/bge-base-en-v1.5",
		"onnx_file": "onnx/model.onnx",
		"pooling_config_path": "1_Pooling",
		"normalize": True
	},
	{
		"model_name": "BAAI/bge-small-en-v1.5",
		"onnx_file": "onnx/model.onnx",
		"pooling_config_path": "1_Pooling",
		"normalize": True
	},
	# jinaai models
	{
		"model_name": "jinaai/jina-embeddings-v2-base-en",
		"onnx_file": "model.onnx",
		"pooling_config_path": "1_Pooling",
		"normalize": False
	},
	{
		"model_name": "jinaai/jina-embeddings-v2-small-en",
		"onnx_file": "model.onnx",
		"pooling_config_path": "1_Pooling",
		"normalize": False
	},
	{
		"model_name": f"{REPO_ORG_NAME}/jina-colbert-v1-en-onnx",
		"base_model": "jinaai/jina-colbert-v1-en",
		"onnx_file": "model.onnx"
	},
	# nomic-ai models
	{
		"model_name": "nomic-ai/nomic-embed-text-v1.5",
		"onnx_file": "onnx/model.onnx",
		"quantized_model_files": {
			"quantized": "onnx/model_quantized.onnx"
		},
		"pooling_config_path": "1_Pooling",
		"normalize": True
	},
	{
		"model_name": "nomic-ai/nomic-embed-text-v1",
		"onnx_file": "onnx/model.onnx",
		"quantized_model_files": {
			"quantized": "onnx/model_quantized.onnx"
		},
		"pooling_config_path": "1_Pooling",
		"normalize": True
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
		
		if len(out_features) == 0:
			error_msg = (
				f"Unable to map ONNX output {self.model_output_names} to either "
				f"`token_embeddings` or `sentence_embedding`. Please define "
				f"the mapping from onnx output names to `token_embeddings` "
				f"or `sentence_embedding`"
			)
			raise ValueError(error_msg)
		
		if "sentence-embedding" not in out_features:
			out_features["attention_mask"] = features["attention_mask"]
		
		return out_features
