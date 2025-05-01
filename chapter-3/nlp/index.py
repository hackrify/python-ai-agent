# Install (if not already installed)
# pip install transformers torch

from transformers import pipeline

# Step 1: Create a summarization pipeline 
summarizer = pipeline("summarization")

# Step 2: Prepare some long text for summarization
text = """
Deep learning has revolutionized the field of artificial intelligence in recent years.
It involves training large neural networks on vast amounts of data to recognize patterns,
make predictions, and uncover insights. Transformers, a breakthrough architecture in
natural language processing, use attention mechanisms to handle long-term dependencies 
efficiently. This has led to state-of-the-art results in tasks like machine translation,
text summarization, and question-answering. Companies worldwide are investing in deep
learning research to enhance their products and services, ultimately shaping the future
of technology and society.
"""

# Step 3: Summarize the text
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

# Step 4: Print the summary
print("Generated Summary:")
print(summary[0]['summary_text'])
