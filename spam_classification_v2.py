import os
import re
import nltk
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import joblib

# For text processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # or WordNetLemmatizer
# from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


def ensure_nltk_resources():
    """
    Ensures that required NLTK resources are downloaded.
    Downloads them only if they are not already present.
    """
    nltk_resources = [
        ("corpora", "stopwords"),
        ("tokenizers", "punkt")
        # If you use WordNetLemmatizer, also add ("corpora", "wordnet")
    ]
    for resource_category, resource_name in nltk_resources:
        try:
            nltk.data.find(f"{resource_category}/{resource_name}")
        except LookupError:
            nltk.download(resource_name)


def download_dataset(url, extract_to):
    """
    Downloads a zip file from the given URL and extracts it to the specified path.

    Parameters
    ----------
    url : str
        The URL of the dataset to download.
    extract_to : str
        The directory to which the ZIP contents will be extracted.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an error for 4XX/5XX codes
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the dataset. Reason: {e}")
        return

    print("Download successful")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(extract_to)
        print("Extraction successful")


def preprocess_message(message, stop_words, stemmer):
    """
    Preprocesses a single message by:
      1. Lowercasing
      2. Removing non-alphabetic characters
      3. Tokenizing
      4. Removing stopwords
      5. Stemming (or lemmatizing)

    Parameters
    ----------
    message : str
        The raw message text to preprocess.
    stop_words : set
        A set of stopwords to remove.
    stemmer : PorterStemmer
        The stemmer to use.

    Returns
    -------
    str
        The preprocessed message as a single string.
    """
    # Lowercase the text
    message = message.lower()

    # Remove everything except alphabetic characters and whitespace
    # If you absolutely need to keep certain symbols, adjust accordingly
    message = re.sub(r"[^a-z\s]", "", message)

    # Tokenize
    tokens = word_tokenize(message)

    # Remove stopwords and short tokens
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    # Optionally lemmatize instead of stemming
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]

    # Join back into a single string
    return " ".join(tokens)


def load_and_preprocess_data(file_path):
    """
    Loads a CSV file containing SMS spam/ham data, preprocesses messages,
    and converts labels to binary (1 for spam, 0 for ham).

    Parameters
    ----------
    file_path : str
        The path to the dataset file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ["label", "message"].
        label: int (1 for spam, 0 for ham)
        message: str (preprocessed text)
    """
    df = pd.read_csv(file_path, sep="\t", header=None, names=["label", "message"])
    df.drop_duplicates(inplace=True)

    # Ensure necessary NLTK data is present
    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    # lemmatizer = WordNetLemmatizer()  # If you choose lemmatization

    df["message"] = df["message"].apply(lambda x: preprocess_message(x, stop_words, stemmer))
    df["label"] = df["label"].apply(lambda x: 1 if x == "spam" else 0)

    return df


def train_model(df):
    """
    Trains a spam detection model using a pipeline of CountVectorizer and MultinomialNB,
    with hyperparameter tuning using GridSearchCV for the alpha parameter.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame, expected to have columns "message" and "label".

    Returns
    -------
    Pipeline
        The best found model (pipeline) according to F1-score cross-validation.
    """
    X = df["message"]
    y = df["label"]

    # Optionally split into train/test to have a final hold-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)

    vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", MultinomialNB())
    ])

    # Hyperparameter grid for the classifier
    param_grid = {
        "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
    }

    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring="f1", 
        n_jobs=-1  # utilize all cores if available
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Best model parameters:", grid_search.best_params_)

    # Evaluate on the hold-out test set
    y_pred = best_model.predict(X_test)
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return best_model


def save_model(model, filename):
    """
    Saves the trained model (pipeline) to a file using joblib.

    Parameters
    ----------
    model : Pipeline
        The scikit-learn pipeline/model to save.
    filename : str
        The filename for the output .joblib file.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Loads a trained scikit-learn pipeline from a joblib file.

    Parameters
    ----------
    filename : str
        The path to the .joblib model file.

    Returns
    -------
    Pipeline
        The loaded pipeline.
    """
    return joblib.load(filename)


def predict_messages(model, messages):
    """
    Predicts whether each message in a list is spam or not spam, 
    printing out the results along with predicted probabilities.

    Parameters
    ----------
    model : Pipeline
        The trained scikit-learn pipeline for spam detection.
    messages : list of str
        The messages to predict.
    """
    predictions = model.predict(messages)
    probabilities = model.predict_proba(messages)

    for i, msg in enumerate(messages):
        prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
        spam_probability = probabilities[i][1]
        ham_probability = probabilities[i][0]

        print(f"Message: {msg}")
        print(f"Prediction: {prediction}")
        print(f"Spam Probability: {spam_probability:.2f}")
        print(f"Not-Spam Probability: {ham_probability:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    # Dataset URL and extraction path
    dataset_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    extract_path = "sms_spam_collection"

    # Download and prepare dataset
    download_dataset(dataset_url, extract_path)
    dataset_path = os.path.join(extract_path, "SMSSpamCollection")
    df = load_and_preprocess_data(dataset_path)

    # Train model
    model = train_model(df)

    # Save model
    save_model(model, "spam_detection_model.joblib")

    # Example usage
    new_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
        "Hey, are we still meeting up for lunch today?",
        "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
        "Reminder: Your appointment is scheduled for tomorrow at 10am.",
        "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
    ]

    # Load and predict
    loaded_model = load_model("spam_detection_model.joblib")
    predict_messages(loaded_model, new_messages)
