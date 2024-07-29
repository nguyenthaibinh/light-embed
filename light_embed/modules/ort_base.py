from typing import Dict, Union
import numpy as np
from pathlib import Path
import onnxruntime
import logging

logger = logging.getLogger(__name__)


class OrtModel:
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
	
	@property
	def model_output_names(self):
		output_names = [
			output_meta.name for output_meta in self._session.get_outputs()
		]
		return output_names
	
	def __call__(
		self,
		features: Dict[str, np.array]
	) -> Dict[str, np.ndarray]:
		ort_output = self._session.run(None, features)

		return ort_output
	
	@classmethod
	def load(cls, input_path: Union[str, Path], **kwargs):
		return cls(input_path, **kwargs)
