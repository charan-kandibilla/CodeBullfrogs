import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv("Phishing_Websites_Data.csv")

# Preprocessing
# One-hot encode categorical features
df = pd.get_dummies(df, columns=["Favicon", "port"])

# Normalize numerical features
scaler = MinMaxScaler()
df[["URL_Length", "age_of_domain", "web_traffic"]] = scaler.fit_transform(df[["URL_Length", "age_of_domain", "web_traffic"]])

# Split the dataset into features (X) and target (y)
X = df.drop("Result", axis=1)
y = df["Result"]

# Map target variable to 0 and 1 for XGBoost compatibility
y = y.map({-1: 0, 1: 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to train and evaluate models
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 60)
    return model

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear'),  # Use 'rbf' for non-linear data
    "XGBoost": XGBClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),  # Adjust n_neighbors as needed
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Train and evaluate each model
trained_models = {}
for name, model in models.items():
    trained_model = train_and_evaluate_model(model, name, X_train, X_test, y_train, y_test)
    trained_models[name] = trained_model

# Compare model accuracies
print("\nModel Comparison:")
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Hyperparameter tuning for XGBoost
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest Parameters for XGBoost:", grid_search.best_params_)
print("Best Accuracy for XGBoost:", grid_search.best_score_)

# Cross-validation for XGBoost
from sklearn.model_selection import cross_val_score

scores = cross_val_score(trained_models["XGBoost"], X, y, cv=5, scoring='accuracy')
print("\nCross-Validation Accuracy for XGBoost:", scores.mean())

# Save the best model (XGBoost)
import joblib
joblib.dump(trained_models["XGBoost"], "best_model.pkl")
print("\nBest model (XGBoost) saved as 'best_model.pkl'")