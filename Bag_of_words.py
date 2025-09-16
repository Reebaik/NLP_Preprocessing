from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown dog barks loudly.",
    "The lazy cat sleeps all day.",
    "Fox and dog are common animals."
]

print("--- Bag-of-Words (BoW) Example ---")
print("\nOriginal Documents:")
for i, doc in enumerate(documents):
    print(f"Doc {i+1}: {doc}")

# 1. Create a CountVectorizer object
# This will tokenize the text and build a vocabulary of known words.
# stop_words='english' removes common words like 'the', 'is', 'a' etc.
vectorizer = CountVectorizer(stop_words='english')

# 2. Fit and Transform the documents
# 'fit' learns the vocabulary from the documents.
# 'transform' converts the documents into a sparse matrix of word counts.
bow_matrix = vectorizer.fit_transform(documents)

# 3. Get the feature names (i.e., the words in our vocabulary)
feature_names = vectorizer.get_feature_names_out()

# 4. Convert the sparse matrix to a dense DataFrame for better visualization
# In real-world scenarios, keep it sparse for efficiency with large datasets
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)

print("\nVocabulary (Feature Names):")
print(feature_names)

print("\nBag-of-Words Matrix (Word Counts):")
print(bow_df)

# Interpretation:
print("\n--- Interpretation ---")
print("Each row represents a document.")
print("Each column represents a unique word from the entire collection of documents (after removing stop words).")
print("The value in each cell indicates how many times that word appears in that specific document.")
print("For example, in 'Doc 1' ('quick brown fox jumps over lazy dog'), 'dog' appears once, 'fox' appears once, etc.")