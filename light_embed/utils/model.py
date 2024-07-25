from typing import Optional, List, Dict, Union
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError

LIGHT_EMBED_NAMESPACE = "LightEmbed"

namespace_map = {
	"sentence-transformers": "sbert",
	"BAAI": "baai",
	"Snowflake": ""
}

def get_onnx_model_dir(
	model_name_or_path: str,
	quantize: bool,
	cache_dir: Optional[str or Path] = None
):
	model_description_json_path = Path(
		model_name_or_path, "model_description.json")
	if model_description_json_path.exists():
		model_dir = model_name_or_path
	else:
		onnx_model_name = get_onnx_model_info(
			base_model_name=model_name_or_path,
			quantize=quantize
		)
		
		try:
			model_dir = download_onnx_model(
				repo_id=onnx_model_name,
				cache_dir=cache_dir
			)
		except RepositoryNotFoundError as _:
			raise ValueError(
				f"Model {model_name_or_path} (quantize={quantize}) "
				f"is not supported in {LIGHT_EMBED_NAMESPACE}."
			)
		except Exception as e:
			raise e
	return model_dir
	

def get_onnx_model_config(
	base_model_name: str,
	quantize: bool,
	supported_models: List[Dict[str, Union[str, Dict[str, str]]]]
) -> Dict[str, Union[str, Dict[str, str]]]:
	quantize_str = str(quantize).lower()
	for model_config in supported_models:
		model_name = model_config.get("model_name", None)
		base_model = model_config.get("base_model", None)
		quantized = model_config.get("quantized", "false")
		if base_model_name in [model_name, base_model] and quantize_str == quantized:
			return model_config
	return None

def download_huggingface_model(
	model_config: Dict,
	cache_dir: Optional[str or Path] = None,
	**kwargs) -> str:

	repo_id = model_config.get("model_name")
	modules_config = model_config.get("modules")

	allow_patterns = [
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"special_tokens_map.json",
		"preprocessor_config.json",
	]

	for module_config in modules_config:
		module_type = module_config.get("type")
		module_path = module_config.get("path")
		
		if module_type == "onnx_model":
			allow_patterns.append(module_path)
		else:
			allow_patterns.append(f"{module_path}/*")
	
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