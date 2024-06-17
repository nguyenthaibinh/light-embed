from typing import Literal, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize(input_array, p=2, dim=1, eps=1e-12) -> np.ndarray:
	# Calculate the Lp norm along the specified dimension
	norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
	norm = np.maximum(norm, eps)  # Avoid division by zero
	normalized_array = input_array / norm
	return normalized_array

def quantize_embeddings(
	embeddings: np.ndarray,
	precision: Literal["float32", "int8", "uint8", "binary", "ubinary"],
	ranges: Optional[np.ndarray] = None,
	calibration_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
	"""
	Quantizes embeddings to a lower precision. This can be used to reduce the memory footprint and increase the
	speed of similarity search. The supported precisions are "float32", "int8", "uint8", "binary", and "ubinary".

	:param embeddings: Unquantized (e.g. float) embeddings with to quantize to a given precision
	:param precision: The precision to convert to. Options are "float32", "int8", "uint8", "binary", "ubinary".
	:param ranges: Ranges for quantization of embeddings. This is only used for int8 quantization, where the ranges
		refers to the minimum and maximum values for each dimension. So, it's a 2D array with shape (2, embedding_dim).
		Default is None, which means that the ranges will be calculated from the calibration embeddings.
	:type ranges: Optional[np.ndarray]
	:param calibration_embeddings: Embeddings used for calibration during quantization. This is only used for int8
		quantization, where the calibration embeddings can be used to compute ranges, i.e. the minimum and maximum
		values for each dimension. Default is None, which means that the ranges will be calculated from the query
		embeddings. This is not recommended.
	:type calibration_embeddings: Optional[np.ndarray]
	:return: Quantized embeddings with the specified precision
	"""
	if isinstance(embeddings, list):
		embeddings = np.array(embeddings)

	if embeddings.dtype in (np.uint8, np.int8):
		raise Exception("Embeddings to quantize must be float rather than int8 or uint8.")

	if precision == "float32":
		return embeddings.astype(np.float32)

	if precision.endswith("int8"):
		# Either use the 1. provided ranges, 2. the calibration dataset or 3. the provided embeddings
		if ranges is None:
			if calibration_embeddings is not None:
				ranges = np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))
			else:
				if embeddings.shape[0] < 100:
					logger.warning(
						f"Computing {precision} quantization buckets based on {len(embeddings)} embedding{'s' if len(embeddings) != 1 else ''}."
						f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
						"or a `calibration_embeddings` that can be used to calculate the buckets."
					)
				ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))
		starts = ranges[0, :]
		steps = (ranges[1, :] - ranges[0, :]) / 255

		if precision == "uint8":
			return ((embeddings - starts) / steps).astype(np.uint8)
		elif precision == "int8":
			return ((embeddings - starts) / steps - 128).astype(np.int8)

	if precision == "binary":
		return (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)

	if precision == "ubinary":
		return np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)

	raise ValueError(f"Precision {precision} is not supported")