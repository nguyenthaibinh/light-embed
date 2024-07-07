from typing import Optional
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
	

def get_onnx_model_info(
	base_model_name: str,
	quantize: bool
):
	namespace, model_id = base_model_name.split("/")
	short_namespace = namespace_map.get(namespace, "")
	
	if namespace == LIGHT_EMBED_NAMESPACE:
		onnx_model_name = base_model_name
	else:
		if short_namespace != "":
			if quantize:
				onnx_model_id = f"{short_namespace}-{model_id}-onnx-quantized"
			else:
				onnx_model_id = f"{short_namespace}-{model_id}-onnx"
		else:
			if quantize:
				onnx_model_id = f"{model_id}-onnx-quantized"
			else:
				onnx_model_id = f"{model_id}-onnx"

		onnx_model_name = f"{LIGHT_EMBED_NAMESPACE}/{onnx_model_id}"

	return onnx_model_name

def download_huggingface_model(
	repo_id: str,
	cache_dir: Optional[str or Path] = None,
	**kwargs) -> str:
	allow_patterns = [
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"special_tokens_map.json",
		"preprocessor_config.json",
		"modules.json",
		"model_description.json",
		"*.onnx",
		"1_Pooling/*"
	]
	
	model_dir = snapshot_download(
		repo_id=repo_id,
		allow_patterns=allow_patterns,
		cache_dir=cache_dir,
		local_files_only=kwargs.get("local_files_only", False),
	)
	return model_dir


def download_onnx_model(
	repo_id: str,
	cache_dir: Optional[str or Path] = None
) -> str:
	model_dir = download_huggingface_model(
		repo_id=repo_id,
		cache_dir=cache_dir
	)
	return model_dir