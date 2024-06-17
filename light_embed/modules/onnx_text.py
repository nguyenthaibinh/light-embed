from typing import Dict, Union
import numpy as np
from pathlib import Path
from .onnx_base import OnnxModel
import logging

logger = logging.getLogger(__name__)

class OnnxText(OnnxModel):
	def __init__(
		self,
		model_path: Union[str, Path],
		**kwargs
	):
		super(OnnxText, self).__init__(model_path, **kwargs)

	def apply(
		self,
		features: Dict[str, np.array]
	):
		ort_output = super().apply(features)
		
		out_features = {
			"token_embeddings": ort_output[0],
			"sentence_embedding": ort_output[-1],
		}
		return out_features
	
	@staticmethod
	def load(input_path: Union[str, Path], **kwargs):
		return OnnxText(input_path, **kwargs)
