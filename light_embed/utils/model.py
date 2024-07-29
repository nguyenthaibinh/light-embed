from typing import Optional, List, Dict, Union
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ORG_NAME = "onnx-models"

def get_managed_model_config(
	base_model_name: str,
	quantize: Union[bool, str],
	managed_models: List[Dict[str, Union[str, Dict[str, str]]]]
) -> Dict[str, Union[str, Dict[str, str]]]:
	quantize_str = str(quantize).lower()
	if quantize_str in ["true", "optimized", "quantized"]:
		quantize_str = "quantized"

	model_config = dict()
	for model_info in managed_models:
		model_name = model_info.get("model_name", None)
		base_model = model_info.get("base_model", None)
		if base_model_name in [model_name, base_model]:
			if quantize_str in ["false", "none"]:
				onnx_file = model_info.get("onnx_file")
			else:
				quantize_path_config = model_info.get("quantized_model_files")
				if quantize_path_config is not None:
					onnx_file = quantize_path_config.get(quantize_str)
			if onnx_file is None:
				return None
			
			model_config["model_name"] = model_name
			model_config["onnx_file"] = onnx_file

			pooling_config_path = model_info.get("pooling_config_path")
			pooling_mode = model_info.get("pooling_mode")
			if isinstance(pooling_config_path, str):
				model_config["pooling_config_path"] = pooling_config_path
			elif isinstance(pooling_mode, str):
				model_config["pooling_mode"] = pooling_mode
					
			model_config["normalize"] = model_info.get("normalize", True)
			return model_config
	return None

def download_huggingface_model(
	model_config: Dict,
	cache_dir: Optional[str or Path] = None,
	**kwargs) -> str:

	repo_id = model_config.get("model_name")

	allow_patterns = [
		model_config.get("onnx_file"),
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"special_tokens_map.json",
		"preprocessor_config.json",
	]
	
	pooling_config_path = model_config.get("pooling_config_path")
	if pooling_config_path is not None:
		allow_patterns.append(f"{pooling_config_path}/*")
	
	model_dir = snapshot_download(
		repo_id=repo_id,
		allow_patterns=allow_patterns,
		cache_dir=cache_dir,
		local_files_only=kwargs.get("local_files_only", False),
	)
	return model_dir


def download_onnx_model(
	model_config: Dict[str, Union[str, Dict[str, str]]],
	cache_dir: Optional[str or Path] = None
) -> str:
	model_dir = download_huggingface_model(
		model_config=model_config,
		cache_dir=cache_dir
	)
	return model_dir