import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data_path = os.path.join('data', 'dataset.csv')  # Adjust the path as necessary
data = pd.read_csv(data_path)

# Split the data into features and labels
X = data['text']
y = data['label']

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# Define parameter grid for GridSearchCV
param_grid = {
    'tfidf__max_df': [0.75, 0.8, 0.85],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__alpha': [0.1, 0.5, 1.0]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

# Output the best parameters and best accuracy score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

# Save the trained model to a file
model_path = 'models/skill_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
joblib.dump(grid_search.best_estimator_, model_path)  # Save the best estimator
print(f"Model saved to {model_path}")
