import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

import pandas as pd

with open('intent_dataset.json', 'r') as file:
    dataset = json.load(file)

examples = []
labels = []
for intent, texts in dataset.items():
    examples.extend(texts)
    labels.extend([intent] * len(texts))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(examples)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, labels)

def classify_intent(text):
    text_vectorized = vectorizer.transform([text])
    confidence_level = classifier.predict_proba(text_vectorized)
    max_confidence = np.max(confidence_level)
    predicted_intent = classifier.predict(text_vectorized)[0]
    
    return predicted_intent, max_confidence

def classify_intent_with_fallback(text, threshold=0.7, fallback_response="NLU fallback: Intent could not be confidently determined"):
    text_vectorized = vectorizer.transform([text])
    confidence_level = classifier.predict_proba(text_vectorized)
    max_confidence = np.max(confidence_level)
    predicted_intent = classifier.predict(text_vectorized)[0]
    
    if max_confidence >= threshold:
        return predicted_intent, max_confidence
    else:
        return fallback_response, max_confidence

phrase = input("Enter a phrase: ")

intent, confidence = classify_intent_with_fallback(phrase)
print("Text: ", phrase)
print("Intent: ", intent)
print("Confidence: ", round(confidence, 2))