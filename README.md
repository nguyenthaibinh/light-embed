LightEmbed is a Light-weight and Fast library for Sentence Transformers model.

Using this model becomes easy when you have LightEmbed installed:

```
pip install -U light-embed
```

Then you can use the model like this:

```python
from light_embed import TextEmbedding
sentences = ["This is an example sentence", "Each sentence is converted"]

model = TextEmbedding('sentence-transformers-model-name')
embeddings = model.encode(sentences)
print(embeddings)
```

For example:
```python
from light_embed import TextEmbedding
sentences = ["This is an example sentence", "Each sentence is converted"]

model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
```

## Citing & Authors

Binh Nguyen / binhcode25@gmail.com