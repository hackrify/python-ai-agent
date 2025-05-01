from transformers import pipeline
translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")

text1 = "I went to the bank to withdraw money."
translation1 = translator(text1)
print(translation1[0]["translation_text"])


text2 = "The boat is tied to the bank of the river."
translation2 = translator(text2)
print(translation1[0]["translation_text"])
