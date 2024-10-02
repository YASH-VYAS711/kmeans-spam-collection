import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Assuming df is the DataFrame loaded from the CSV file
df = pd.read_csv('/kaggle/input/spamdata/spam.csv', encoding='latin-1',usecols=[0,1])
# Display the first few rows to understand the structure
print(df.head())
# Rename columns if necessary (depending on the actual column names in your CSV)
df.columns = ['label', 'text']
# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
# Encode labels to numeric values
y = y.map({'ham': 0, 'spam': 1})
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Make predictions
y_pred = knn.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
