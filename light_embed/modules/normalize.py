from typing import Dict
import numpy as np
from pathlib import Path


class Normalize:
	"""
	This layer normalizes embeddings to unit length
	"""

	def __init__(self):
		super(Normalize, self).__init__()
		
	@staticmethod
	def __call__(
		features: Dict[str, np.array],
		p=2, dim=1, eps=1e-12) -> Dict[str, np.ndarray]:
		# Calculate the Lp norm along the specified dimension
		input_array = features.get("sentence_embedding")
		norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
		norm = np.maximum(norm, eps)  # Avoid division by zero
		normalized_array = input_array / norm
		features.update({"sentence_embedding": normalized_array})
		return features

	def save(self, output_path):
		pass

	@staticmethod
	def load(input_path: str or Path):
		return Normalize()