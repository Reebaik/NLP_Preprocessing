from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents (same as before for comparison)
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown dog barks loudly.",
    "The lazy cat sleeps all day.",
    "Fox and dog are common animals."
]

print("--- TF-IDF Example ---")
print("\nOriginal Documents:")
for i, doc in enumerate(documents):
    print(f"Doc {i+1}: {doc}")

# 1. Create a TfidfVectorizer object
# This will tokenize the text, build a vocabulary, and compute TF-IDF scores.
# stop_words='english' removes common words.
vectorizer = TfidfVectorizer(stop_words='english')

# 2. Fit and Transform the documents
# 'fit' learns the vocabulary and IDF values from the documents.
# 'transform' converts the documents into a sparse matrix of TF-IDF scores.
tfidf_matrix = vectorizer.fit_transform(documents)

# 3. Get the feature names (i.e., the words in our vocabulary)
feature_names = vectorizer.get_feature_names_out()

# 4. Convert the sparse matrix to a dense DataFrame for better visualization
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("\nVocabulary (Feature Names):")
print(feature_names)

print("\nTF-IDF Matrix:")
print(tfidf_df)

# Interpretation:
print("\n--- Interpretation ---")
print("Each row represents a document.")
print("Each column represents a unique word from the entire collection of documents (after removing stop words).")
print("The value in each cell is the TF-IDF score for that word in that specific document.")
print("Higher TF-IDF scores indicate words that are important to that document (appear frequently in it) ")
