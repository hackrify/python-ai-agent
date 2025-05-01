from transformers import pipeline
translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
text = "Hello, how are you?"
translation = translator(text)
print(translation[0]["translation_text"])
