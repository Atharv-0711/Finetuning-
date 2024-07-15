# Finetuning-
import os
import json
import keras
import keras_nlp
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
!pip install -q -U keras-nlp
!pip install -q -U keras>=3
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
!wget -O databricks-dolly-15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
import json
data = []
with open("databricks-dolly-15k.jsonl") as file:
    for line in file:
        features = json.loads(line)
        if not features["context"]:
            template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
            data.append(template.format(**features))

# Use only the first 1000 examples for faster training
data = data[:1000]
import keras_nlp
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()
prompt1 = "What should I do on a trip to Europe?"
prompt2 = "Explain the process of photosynthesis in a way that a child could understand."

sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt1, max_length=256))
print(gemma_lm.generate(prompt2, max_length=256))

gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()
gemma_lm.preprocessor.sequence_length = 512

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

gemma_lm.fit(data, epochs=1, batch_size=1)

print(gemma_lm.generate(prompt1, max_length=256))
print(gemma_lm.generate(prompt2, max_length=256))
# Define prompt for travel to Germany
prompt = "How will I travel to Germany?"

# Compile model
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)

# Generate response
response = gemma_lm.generate(prompt, max_length=256)

# Print generated response
print("Response to 'How will I travel to Germany?':")
print(response)
