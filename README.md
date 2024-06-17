# LightEmbed

LightEmbed is a light-weight, fast, and efficient tool for generating sentence embeddings. It does not rely on heavy dependencies like PyTorch and Transformers, making it suitable for environments with limited resources.

## Benefits

#### 1. Light-weight
- **Minimal Dependencies**: LightEmbed does not depend on PyTorch and Transformers.
- **Low Resource Requirements**: Operates smoothly with minimal specs: 1GB RAM, 1 CPU, and no GPU required.

#### 2. Fast (as light)
- **ONNX Runtime**: Utilizes the ONNX runtime, which is significantly faster compared to Sentence Transformers that use PyTorch.

#### 3. Same as Original Sentence Transformers' Outputs
- **Consistency**: Incorporates all modules from a Sentence Transformer model, including normalization and pooling.
- **Accuracy**: Produces embedding vectors identical to those from Sentence Transformers.

## Installation
```
pip install -U light-embed
```

## Usage
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