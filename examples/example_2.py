import os
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
		default="jinaai/jina-embeddings-v2-base-en"
	)
	parser.add_argument(
		"--onnx-file", type=str, default="model.onnx",
		help="Relative path to the onnx file"
	)
	parser.add_argument(
		"--normalize-embeddings", default=False, action="store_true"
	)
	args = parser.parse_args()
	model_name = args.model_name
	onnx_file = args.onnx_file
	normalize_embeddings = args.normalize_embeddings
	
	load_dotenv()
	
	cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

	embedding_model = TextEmbedding(
		model_name=model_name,
		onnx_file=onnx_file,
		pooling_config_path="1_Pooling",
		normalize=False,
		cache_folder=cache_dir,
		# device="cpu"
	)
	
	print("embedding_model:", embedding_model)
	
	sentences = [
		(
			"Sentence embeddings are a technique in natural language processing where "
			"sentences are converted into fixed-size vectors. These embeddings capture "
			"the semantic meaning of sentences, allowing for effective comparison "
			"and retrieval in various NLP tasks such as text similarity, classification, "
			"and information retrieval."
		),
		(
			"Embeddings represent sentences as dense vectors in a continuous space, "
			"capturing their semantic properties. By encoding sentences into these "
			"vector representations, we can perform tasks like measuring similarity, "
			"clustering, and searching in a way that reflects the underlying "
			"meaning and context of the text."
		)
	]
	
	start_embed = timer()
	embeddings = embedding_model.encode(
		sentences, output_value="sentence_embedding",
		normalize_embeddings=normalize_embeddings,
		return_as_array=False,
		return_as_list=True
	)
	elapsed_time = timer() - start_embed
	
	print("embeddings:\n", embeddings)
	print("model_name:", model_name)
	print("embedding dimension:", len(embeddings[0]))
	print(f"elapsed time: {elapsed_time:.2f}")
	
	return None


if __name__ == "__main__":
	main()