from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
prompt = "Once upon a time"
output = generator(prompt, max_length=50)
print(output[0]["generated_text"])
