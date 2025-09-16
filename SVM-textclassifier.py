from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset
texts = [
    'I love this movie',
    'This movie is terrible',
    'I really enjoyed this film',
    'This film is awful',
    'What a fantastic experience',
    'I hated this film',
    'This was a great movie',
    'The film was not good',
    'I am very happy with this movie'
]

# Corresponding labels (1 = positive, 0 = negative)
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1]

# Split data into training and testing (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Create pipeline with CountVectorizer and SVM
model = make_pipeline(CountVectorizer(), SVC(kernel='linear'))

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Accuracy score
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Test the final sentence separately
test_sentence = ['I am disappointed with this film']
pred = model.predict(test_sentence)
print(f"\nTest Sentence: {test_sentence[0]}")
print(f"Predicted Sentiment: {'Positive' if pred[0] == 1 else 'Negative'}")
