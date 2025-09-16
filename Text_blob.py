from textblob import TextBlob

# The text we want to analyze
text = "TextBlob is a wonderful tool for basic natural language processing tasks. It's incredibly easy to use and provides quick insights."

# Create a TextBlob object
blob = TextBlob(text)

print(f"--- Analyzing Text ---\n'{text}'\n")

# 1. Part-of-Speech (POS) Tagging
print("--- Part-of-Speech Tagging ---")
for word, tag in blob.tags:
    print(f"Word: '{word}', Tag: '{tag}'")
print("-" * 30)

# 2. Noun Phrase Extraction
print("\n--- Noun Phrase Extraction ---")
if blob.noun_phrases:
    for phrase in blob.noun_phrases:
        print(f"Noun Phrase: '{phrase}'")
else:
    print("No noun phrases found.")
print("-" * 30)

# 3. Sentiment Analysis
print("\n--- Sentiment Analysis ---")
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

sentiment_label = "Neutral"
if polarity > 0.05:
    sentiment_label = "Positive"
elif polarity < -0.05:
    sentiment_label = "Negative"

print(f"Polarity (range -1.0 to 1.0): {polarity:.2f}")
print(f"Subjectivity (range 0.0 to 1.0): {subjectivity:.2f}")
print(f"Overall Sentiment: {sentiment_label}")
print("-" * 30)