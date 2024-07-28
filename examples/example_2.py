import os
from typing import List
import light_embed
from light_embed import TextEmbedding
from dotenv import load_dotenv
from timeit import default_timer as timer


def main():
	print(f"light_embed.__version__: {light_embed.__version__}")
	
	model_name = "BAAI/bge-base-en"
	
	load_dotenv()
	
	cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")
	
	modules_config = [
		{
			"type": "onnx_model",
			"path": "onnx/model.onnx"
		},
		{
			"type": "pooling",
			"path": "1_Pooling"
		},
		{
			"type": "normalize",
			"path": "2_Normalize"
		}
	]

	embedding_model = TextEmbedding(
		model_name_or_path=model_name,
		cache_folder=cache_dir,
		quantize=False,
		device="cpu",
		modules_config=modules_config
	)
	
	print("embedding_model:", embedding_model)
	
	documents: List[str] = [
		"This is a light-weight and fast python library for generating embeddings.",
		"This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
	]
	
	start_embed = timer()
	embeddings = embedding_model.encode(
		documents, output_value="sentence_embedding",
		return_as_array=False,
		return_as_list=True
	)
	elapsed_time = timer() - start_embed
	
	print("embeddings:\n", embeddings)
	print("type(embeddings):", type(embeddings))
	
	print("len(embeddings[0]):", len(embeddings[0]))
	print("type(embeddings[0]):", type(embeddings[0]))
	print("model_name:", model_name)
	print(f"elapsed time: {elapsed_time:.2f}")
	
	return None


if __name__ == "__main__":
	main()