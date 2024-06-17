from typing import Dict, Union
import numpy as np
from pathlib import Path
import onnxruntime
import psutil
import logging

logger = logging.getLogger(__name__)


class OnnxModel:
	device = "cpu"
	
	def __init__(
		self,
		model_path: Union[str, Path],
		**kwargs
	):
		onnxproviders = onnxruntime.get_available_providers()
		
		self.device = kwargs.pop("device", self.device)
		
		if self.device == "cpu":
			fast_onnxprovider = "CPUExecutionProvider"
		else:
			if "CUDAExecutionProvider" not in onnxproviders:
				logger.warning("Using CPU. Try installing 'onnxruntime-gpu'.")
				fast_onnxprovider = "CPUExecutionProvider"
			else:
				fast_onnxprovider = "CUDAExecutionProvider"
		
		sess_options = onnxruntime.SessionOptions()
		sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
		sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
		
		self._session = onnxruntime.InferenceSession(
			str(model_path), sess_options,
			providers=[fast_onnxprovider]
		)
	
	@property
	def model_input_names(self):
		input_names = [
			input_meta.name for input_meta in self._session.get_inputs()
		]
		return input_names
	
	def apply(
		self,
		features: Dict[str, np.array]
	):
		ort_output = self._session.run(None, features)
		return ort_output
	
	@staticmethod
	def load(input_path: Union[str, Path], **kwargs):
		return OnnxModel(input_path, **kwargs)
