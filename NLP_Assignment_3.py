import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Sample data
data = {'sentence': ["This is a sample sentence.",
                     "Dr. Smith attended the conference.",
                     "The price increased by 10%. How will it affect us?",
                     "Please submit the report by Friday!",
                     "Inc. is a common abbreviation.",
                     "The project deadline is approaching!",
                     "This is an example Dr. of a sentence.",
                     "How are you today?",
                     "This is a test sentence with numbers like .02% or 4.3."]}

df = pd.DataFrame(data)

# Function to preprocess the text
def preprocess_text(text):
    # Remove unwanted characters
    text = re.sub(r'[^a-zA-Z0-9\s\.\?!%]', '', text)

    # Handle specific cases
    text = re.sub(r'\b(?:Dr\.|Inc\.|Mr\.|Mrs\.|Ms\.)\b', 'ABBREV', text)  # Replace common abbreviations
    text = re.sub(r'\d+(\.\d+)?%?', 'NUM', text)  # Replace numbers and percentages

    return text.lower()

# Apply preprocessing to the sentences
df['processed_sentence'] = df['sentence'].apply(preprocess_text)

# Label the sentences manually (1 for end-of-sentence, 0 for not)
df['is_end_of_sentence'] = [0, 1, 1, 1, 1, 1, 0, 1, 1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_sentence'], df['is_end_of_sentence'], test_size=0.2, random_state=42
)

# Create pipelines for different classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machine': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Neural Network': MLPClassifier()
}

for classifier_name, classifier in classifiers.items():
    # Create a pipeline with CountVectorizer and the current classifier
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', classifier)
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"{classifier_name} Model Accuracy: {accuracy * 100:.2f}%\n")

    # Test the model on new sentences
    new_sentences = ["The project deadline is approaching.",
                     "This is an example Dr. of a sentence.",
                     "How are you today?",
                     "This is a test sentence with numbers like .02% or 4.3.",
                     "What time is the meeting at 2.30 PM? It's important."]

    new_sentences_processed = [preprocess_text(sentence) for sentence in new_sentences]
    new_predictions = model.predict(new_sentences_processed)

    print(f"{classifier_name} Predictions:")
    for sentence, prediction in zip(new_sentences, new_predictions):
        print(f"\nSentence: {sentence}\nPrediction: {'End of Sentence' if prediction == 1 else 'Not End of Sentence'}\n")

    # Print the model details at the end
    print(f"\n{classifier_name} Model Used:")
    print(model)
    print("\n" + "="*50 + "\n")
