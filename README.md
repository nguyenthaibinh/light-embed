# LightEmbed

LightEmbed is a light-weight, fast, and efficient tool for generating sentence embeddings. It does not rely on heavy dependencies like PyTorch and Transformers, making it suitable for environments with limited resources.

## Benefits

#### 1. Light-weight
- **Minimal Dependencies**: LightEmbed does not depend on PyTorch and Transformers.
- **Low Resource Requirements**: Operates smoothly with minimal specs: 1GB RAM, 1 CPU, and no GPU required.

#### 2. Fast (as light)
- **ONNX Runtime**: Utilizes the ONNX runtime, which is significantly faster compared to Sentence Transformers that use PyTorch.

#### 3. Consistent with Sentence Transformers
- **Consistency**: Incorporates all modules from a Sentence Transformer model, including normalization and pooling.
- **Accuracy**: Produces embedding vectors identical to those from Sentence Transformers.

#### 4. Supports models not managed by LightEmbed
LightEmbed can work with any Hugging Face repository, even those not hosted on 
[Hugging Face ONNX models](https://huggingface.co/onnx-models), as long as ONNX files are available.

#### 5. Local Model Support
LightEmbed can load models from the local file system, enabling faster loading times and functionality
in environments without internet access, such as AWS Lambda or EC2 instances in private subnets.


## Installation
```
pip install -U light-embed
```

## Usage

Then you can specify the `original model name` like this:
```python
from light_embed import TextEmbedding
sentences = ["This is an example sentence", "Each sentence is converted"]

model = TextEmbedding(model_name_or_path='sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
```

or, alternatively, you can specify the `onnx model name` like this:
```python
from light_embed import TextEmbedding
sentences = ["This is an example sentence", "Each sentence is converted"]

model = TextEmbedding(model_name_or_path='onnx-models/all-MiniLM-L6-v2-onnx')
embeddings = model.encode(sentences)
print(embeddings)
```

If the model is not on [Hugging Face ONNX models](https://huggingface.co/onnx-models),
you can specify model name and the model_config:
```python
from light_embed import TextEmbedding
sentences = ["This is an example sentence", "Each sentence is converted"]

model_config = {
    "model_file": "model.onnx",
    "pooling_config_path": "1_Pooling",
    "normalize": False
}
model = TextEmbedding(
    model_name_or_path='onnx-models/all-MiniLM-L6-v2-onnx',
    model_config=model_config
)
embeddings = model.encode(sentences)
print(embeddings)
```

**Use local model**: to use the local model, you need to specify the path to the folder and the model_config:
```python
from light_embed import TextEmbedding
sentences = ["This is an example sentence", "Each sentence is converted"]

model_config = {
    "model_file": "model.onnx",
    "pooling_config_path": "1_Pooling",
    "normalize": False
}
model = TextEmbedding(
    model_name_or_path='/path/to/the/local/model/all-MiniLM-L6-v2-onnx',
    model_config=model_config
)
embeddings = model.encode(sentences)
print(embeddings)
```

## Citing & Authors

Binh Nguyen / binhcode25@gmail.com