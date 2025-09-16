import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Example corpus
documents = [
    "The cat sat on the mat",
    "Dogs and cats are friends",
    "I love to play football",
    "The game of cricket is popular in India",
    "Python and Java are programming languages",
    "Machine learning and AI are future technologies"
]

# Tokenize
texts = [doc.lower().split() for doc in documents]

# Dictionary & Corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Output folders
os.makedirs("wordclouds", exist_ok=True)
os.makedirs("pyldavis_html", exist_ok=True)

# Train LDA with 2 topics
num_topics = 2
lda_model = models.LdaModel(
    corpus, num_topics=num_topics, id2word=dictionary,
    passes=10, random_state=42
)

# pyLDAvis HTML
vis = gensimvis.prepare(lda_model, corpus, dictionary)
html_path = f"pyldavis_html/topics_{num_topics}.html"
pyLDAvis.save_html(vis, html_path)
print(f"✅ pyLDAvis saved: {html_path}")

# Word Clouds (one per topic)
for t in range(num_topics):
    wc = WordCloud(background_color='white', width=800, height=400)
    wc.generate_from_frequencies(dict(lda_model.show_topic(t, 30)))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"wordclouds/topic_{t}_k{num_topics}.png", dpi=150)
    plt.close()
