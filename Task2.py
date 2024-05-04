import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

import pandas as pd

with open('intent_dataset.json', 'r') as file:
    dataset = json.load(file)

# Extract examples and labels from the loaded dataset
examples = []
labels = []
for intent, texts in dataset.items():
    examples.extend(texts)
    labels.extend([intent] * len(texts))

# Model training
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(examples)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, labels)

# Intent classification function
def classify_intent(text):
    text_vec = vectorizer.transform([text])
    confidence_scores = classifier.predict_proba(text_vec)
    max_confidence = np.max(confidence_scores)
    predicted_intent = classifier.predict(text_vec)[0]
    
    return predicted_intent, max_confidence

# Intent classification function with fallback mechanism
def classify_intent_with_fallback(text, threshold=0.7, fallback_response="NLU fallback: Intent could not be confidently determined"):
    text_vec = vectorizer.transform([text])
    confidence_scores = classifier.predict_proba(text_vec)
    max_confidence = np.max(confidence_scores)
    predicted_intent = classifier.predict(text_vec)[0]
    
    if max_confidence >= threshold:
        return predicted_intent, max_confidence
    else:
        return fallback_response, max_confidence

# Getting user input
phrase = input("Enter a phrase: ")

intent, confidence = classify_intent_with_fallback(phrase)
print("Text: ", phrase)
print("Intent: ", intent)
print("Confidence: ", confidence)