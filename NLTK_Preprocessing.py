import nltk
from nltk.tokenize import word_tokenize, sent_tokenize  # Added sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text
text = "Torrential Rains innudate New York Citys. Brings Chaos to subways and streets."

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentence Tokenization:")
for i, sent in enumerate(sentences, 1):
    print(f"  {i}. {sent}")

# Word Tokenization
tokens = word_tokenize(text)
print("\nWord Tokens:", tokens)

# Stop word removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("After Stop Word Removal:", filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]
print("After Stemming:", stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("After Lemmatization:", lemmatized)
