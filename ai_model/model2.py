import pandas as pd
import pickle
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Load dataset
dataset_path = "webpage_text_dataset.csv"
df = pd.read_csv(dataset_path)

df.dropna(inplace=True)  # Remove missing values

# Convert labels to numerical format
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train XGBoost model with tuned parameters
model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    max_depth=6, 
    learning_rate=0.1, 
    n_estimators=200, 
    scale_pos_weight=1.5  # Adjust weight for class imbalance
)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate model
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Save the model and vectorizer
with open("text_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")