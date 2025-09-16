import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Sentence to analyze
text = "Microsoft is owned by Bill Gates"

# Process the sentence
doc = nlp(text)

# Display token-wise details in a clean table
print(f"{'Token':<12}{'Dependency':<15}{'Head':<12}{'POS'}")
print("-" * 50)
for token in doc:
    print(f"{token.text:<12}{token.dep_:<15}{token.head.text:<12}{token.pos_}")

# Extracting relationships
subject = ""
object_ = ""
relation = ""

for token in doc:
    if token.dep_ == "nsubjpass":   # Passive subject (Microsoft)
        subject = token.text
    elif token.dep_ == "agent":     # Agent of passive verb
        object_ = " ".join(child.text for child in token.children if child.dep_ == "pobj")
    elif token.dep_ == "ROOT":      # Main verb
        relation = token.text

# Nicely structured relationship output
print("\nExtracted Relationship:")
print(f"{subject} <-- {relation} -- {object_}")
