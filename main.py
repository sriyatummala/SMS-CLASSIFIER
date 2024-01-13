import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    # Load dataset
    df_sms = pd.read_csv('spam.csv', encoding='latin-1')
    df_sms = df_sms[['v1', 'v2']]
    df_sms.columns = ['label', 'message']

    # Text preprocessing and vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_sms['message'])
    y = df_sms['label']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model \
    predictions = model.predict(X_test)

    # Calculate the classification metrics
    report = classification_report(y_test, predictions, output_dict=True)

    # Extracting overall accuracy and other metrics for each class
    accuracy = report['accuracy']
    precision_class_0 = report['ham']['precision']
    recall_class_0 = report['ham']['recall']
    f1_score_class_0 = report['ham']['f1-score']

    precision_class_1 = report['spam']['precision']
    recall_class_1 = report['spam']['recall']
    f1_score_class_1 = report['spam']['f1-score']

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)

    # Evaluation results
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision (Class 0): {:.2f}%".format(precision_class_0 * 100))
    print("Recall (Class 0): {:.2f}%".format(recall_class_0 * 100))
    print("F1 Score (Class 0): {:.2f}%".format(f1_score_class_0 * 100))
    print("Precision (Class 1): {:.2f}%".format(precision_class_1 * 100))
    print("Recall (Class 1): {:.2f}%".format(recall_class_1 * 100))
    print("F1 Score (Class 1): {:.2f}%".format(f1_score_class_1 * 100))
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_report(y_test, predictions))

