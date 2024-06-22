from typing import Optional, Dict, Any
from pathlib import Path
from huggingface_hub import snapshot_download

LIGHT_EMBED_ORG_NAME = "LightEmbed"

org_name_map = {
	"sentence-transformers": "sbert",
	"BAAI": "baai",
	"Snowflake": ""
}

def get_onnx_model_info(
	base_model_name: str,
	quantize: bool
):
	org_name, model_suffix = base_model_name.split("/")
	org_short_name = org_name_map.get(org_name, "")
	
	if org_short_name == LIGHT_EMBED_ORG_NAME:
		onnx_model_name = base_model_name
	else:
		if org_short_name != "":
			if quantize:
				onnx_model_suffix = f"{org_short_name}-{model_suffix}-onnx-quantized"
			else:
				onnx_model_suffix = f"{org_short_name}-{model_suffix}-onnx"
		else:
			if quantize:
				onnx_model_suffix = f"{model_suffix}-onnx-quantized"
			else:
				onnx_model_suffix = f"{model_suffix}-onnx"

		onnx_model_name = f"{LIGHT_EMBED_ORG_NAME}/{onnx_model_suffix}"
		
	model_info = {
		"model_name": onnx_model_name,
		"base_model": base_model_name,
		"quantize": str(quantize),
		"model_file": "model.onnx"
	}
	return model_info

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