import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Torrential Rains innudate New York Citys.Brings Chaos to subways and streets"

# Process the text
doc = nlp(text)

# Sentence Tokenization
print("Sentence Tokenization:")
for i, sent in enumerate(doc.sents, 1):
    print(f"  {i}. {sent.text}")

# Word Tokenization
tokens = [token.text for token in doc]
print("\nWord Tokens:", tokens)

# Stop word removal
filtered_tokens = [token for token in doc if not token.is_stop and token.is_alpha]
print("After Stop Word Removal:", [token.text for token in filtered_tokens])

# Lemmatization (after stop word removal)
lemmatized = [token.lemma_ for token in filtered_tokens]
print("After Lemmatization:", lemmatized)

# Note: spaCy doesn't do stemming directly (it's lemmatization-focused)
