import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import seaborn as sns
import math

# Sample documents (each as a paragraph)
docs = [
    """Cats are independent, curious animals often kept as pets.
    They are known for their agility, playful behavior, and hunting instincts.
    Many people enjoy the companionship of cats due to their low maintenance.""",

    """Dogs are loyal companions and are often called man's best friend.
    They have been domesticated for thousands of years for protection, companionship, and work.
    Dogs can be trained for a variety of tasks, including guiding the visually impaired.""",

    """Birds are fascinating creatures with the ability to fly.
    They are found in diverse environments across the globe.
    Some birds are known for their vibrant plumage and melodic songs.""",

    """Cats and dogs, despite their differences, often live together in harmony.
    Many households raise both animals, observing their unique interactions.
    The bond between pets and humans can be incredibly enriching and joyful."""
]

# Step 1: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(docs)
feature_names = vectorizer.get_feature_names_out()

# Step 2: Generate Word Clouds for Each Document
n_docs = len(docs)
n_cols = 2  # You can change this for wider layouts
n_rows = math.ceil(n_docs / n_cols)

# Use constrained_layout for better spacing
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 6 * n_rows), constrained_layout=True)
axes = axes.flatten()

for i in range(n_docs):
    tfidf_scores = tfidf_matrix[i].toarray().flatten()
    doc_word_scores = {
        word: tfidf_scores[j]
        for j, word in enumerate(feature_names)
        if tfidf_scores[j] > 0
    }

    wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(doc_word_scores)
    axes[i].imshow(wordcloud, interpolation="bilinear")
    axes[i].axis("off")
    axes[i].set_title(f"Word Cloud for Document {i+1}", fontsize=14, pad=10)

# Hide unused axes if any
for j in range(n_docs, len(axes)):
    axes[j].axis("off")

plt.show()

# Step 3: Cosine Similarity Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cosine_similarity(tfidf_matrix), annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=[f'Doc {i+1}' for i in range(n_docs)],
            yticklabels=[f'Doc {i+1}' for i in range(n_docs)])
plt.title("TF-IDF Cosine Similarity Matrix", fontsize=14)
plt.tight_layout()
plt.show()
