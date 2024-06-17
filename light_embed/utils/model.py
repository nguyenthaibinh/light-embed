from typing import Optional, Dict, Any
from pathlib import Path
import json
from huggingface_hub import snapshot_download

def download_model_from_huggingface(
	model_name: str,
	cache_dir: Optional[str or Path] = None,
	**kwargs) -> str:
	allow_patterns = [
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"special_tokens_map.json",
		"preprocessor_config.json",
		"modules.json",
		"*.onnx",
		"1_Pooling/*"
	]
	
	model_dir = snapshot_download(
		repo_id=model_name,
		allow_patterns=allow_patterns,
		cache_dir=cache_dir,
		local_files_only=kwargs.get("local_files_only", False),
	)
	return model_dir


def download_onnx_model(
	model_info: Dict[str, Any],
	cache_dir: Optional[str or Path] = None
) -> str:
	model_name = model_info["model_name"]
	model_dir = download_model_from_huggingface(
		model_name=model_name,
		cache_dir=cache_dir
	)
	return model_dir

def get_onnx_model_info(
	model_name: str,
	quantize: bool
) -> Dict[str, str]:
	current_dir = Path(__file__).parent
	# Load the modules of sentence transformer
	supported_models_json_path = Path(current_dir, "supported_models.json")
	with open(supported_models_json_path) as fIn:
		supported_models = json.load(fIn)
		
	quantize_str = "true" if quantize else "false"
	
	for model_info in supported_models:
		if model_name in (model_info["model_name"], model_info["base_model"]):
			if quantize_str == model_info.get("quantized"):
				return model_info
			else:
				break
	return None