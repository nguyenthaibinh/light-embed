import os
import numpy as np
import light_embed
from light_embed import TextEmbedding
from timeit import default_timer as timer

def main():
	print(f"light_embed.__version__: {light_embed.__version__}")

	model_name_or_path = "jinaai/jina-embeddings-v3"
	embedding_model = TextEmbedding(model_name_or_path='jinaai/jina-embeddings-v3', device="cpu")
	config_lora_adaptations = embedding_model._transformer_config.get("lora_adaptations")
	
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

	task_type = 'retrieval.passage'
	task_id = np.array(config_lora_adaptations.index(task_type), dtype=np.int64)
	start_embed = timer()
	embeddings = embedding_model.encode(
		sentences, output_value="sentence_embedding",
		extra_kwargs={"task_id":task_id},
		return_as_array=False,
		return_as_list=True
	)
	elapsed_time = timer() - start_embed
	
	print("embeddings:\n", embeddings)
	print("model_name_or_path:", model_name_or_path)
	print("embedding dimension:", len(embeddings[0]))
	print(f"elapsed time: {elapsed_time:.2f}")
	
	return None

if __name__ == "__main__":
	main()
