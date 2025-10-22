import requests
from bs4 import BeautifulSoup
import pickle
import re
import numpy as np

# Load trained models
vectorizer_path = "ai_model/vectorizer.pkl"
model_path = "ai_model/text_model.pkl"

with open(vectorizer_path, "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open(model_path, "rb") as model_file:
    text_model = pickle.load(model_file)

def extract_text_from_url(url):
    """
    Extracts visible text from a webpage URL.

    :param url: URL of the webpage.
    :return: Extracted text (cleaned) or an error message dictionary.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            return {"error": f"Failed to fetch webpage. Status code: {response.status_code}"}

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Extract visible text
        text = soup.get_text(separator=" ").strip()

        # Ensure the extracted text is a string before processing
        if not isinstance(text, str) or len(text) == 0:
            return {"error": "Invalid or empty text extracted"}

        # Clean the text
        text = re.sub(r"\s+", " ", text)  # Normalize spaces
        text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
        return text.lower()

    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    

def classify_webpage_text(url):
    """
    Classifies the webpage content as phishing or legitimate.
    
    :param url: URL of the webpage.
    :return: Classification result.
    """
    text = extract_text_from_url(url)

    if isinstance(text, dict) and "error" in text:
        return text  # Return error message if extraction fails

    text_vectorized = vectorizer.transform([text])  # Convert text to numerical form
    prediction = text_model.predict(text_vectorized)[0]  # Get model prediction

    result = "phishing" if prediction == 1 else "legitimate"

    return {
        "url": url,
        "classification": result,
        "message": f"The webpage is classified as {result}."
    }
