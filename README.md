Phishing Website Detection Browser Extension

Overview

This project is a phishing site detection system that integrates machine learning (ML) models into a browser extension to detect malicious websites. It uses:

URL-based feature analysis (XGBoost model trained on phishing datasets)

Webpage content analysis (TF-IDF vectorized text classification)

SSL certificate validation

Behavioral analysis (detecting suspicious user interactions)

Check out the document "CodeBullfrogs" in the main branch for more info on the project and output screenshots.

Features

➢ Behavioural Analysis of Websites
➢ AI-Based Content (text), URL, and SSL Analysis
➢ Multi-Layered AI Detection
➢ Multi-Layered Security
➢ Real-Time Browser Extension

Features of model.py and model2.py which are the 2 ML models:

model.py (Phishing Detection Using Multiple ML Models):
- The script loads a dataset containing phishing-related features, preprocesses the data by encoding categorical variables, and normalizes numerical features.
  
- It trains and evaluates multiple machine learning models, including Random Forest, Logistic Regression, SVM, XGBoost, KNN, and Neural Networks, to classify websites as phishing or legitimate.
  
- After model comparison, it fine-tunes the XGBoost classifier using GridSearchCV and cross-validation, saving the best-performing model as best_model.pkl.

model2.py (Phishing Detection Based on Webpage Text):
- This script focuses on phishing detection using textual content by extracting and vectorizing webpage text with TF-IDF and handling class imbalance using SMOTE.
  
- It trains an XGBoost classifier on the text-based features, optimizing it with hyperparameters such as max_depth, learning_rate, and n_estimators.
  
- The trained model is evaluated using accuracy, precision, and recall metrics, and the final model along with the vectorizer is saved as text_model.pkl and vectorizer.pkl.

Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/aditi-acharya/CodeBullfrogs.git
cd your-repo

2️⃣ Install Dependencies

Make sure you have Python installed, then run:

pip install -r requirements.txt

3️⃣ Train the Model (Optional)

If needed, retrain the text-based ML model:

python ai_model/model2.py
python ai_model/model.py

4️⃣ Run the Flask Backend

python app.py

5️⃣ Load the Chrome Extension

Open Chrome and go to chrome://extensions/.

Enable Developer Mode (top right corner).

Click Load Unpacked and select the browser_extension folder.

Usage

When browsing, the extension will automatically check each visited website.

If a site is classified as phishing, you can check on service wroker (of chrome extension) that it is classified as such.

The extension also logs user interactions for behavioral analysis.

Contributors: Aditi P Acharya, K Charan Kandibilla
