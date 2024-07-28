import os
from typing import List
import light_embed
from light_embed import TextEmbedding
from dotenv import load_dotenv
from timeit import default_timer as timer
import argparse

def main():
	print(f"light_embed.__version__: {light_embed.__version__}")

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--model-name", type=str,
		default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
	)
	parser.add_argument(
		"--normalize-embeddings", default=False, action="store_true"
	)
	args = parser.parse_args()
	model_name = args.model_name
	normalize_embeddings = args.normalize_embeddings

	load_dotenv()

	cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")
	
	embedding_model = TextEmbedding(
		model_name_or_path=model_name,
		cache_folder=cache_dir,
		quantize=False,
		device="cpu"
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
		return_as_list=True,
		normalize_embeddings=normalize_embeddings
	)
	elapsed_time = timer() - start_embed
	
	print("embeddings:\n", embeddings)
	print("model_name:", model_name)
	print("embedding dimension:", len(embeddings[0]))
	print(f"elapsed time: {elapsed_time:.2f}")
	
	return None

if __name__ == "__main__":
	main()