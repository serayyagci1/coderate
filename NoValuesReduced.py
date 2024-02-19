import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
address = "heart_2022_cleared.csv"
df = pd.read_csv(address)

# Define the column name and value for filtering
column_name = 'HeartDisease'
value_to_drop = 'No'

# Identify the indices of rows to drop based on the specified condition
rows_to_drop_indices = df[df[column_name] == value_to_drop].sample(frac=0.8, random_state=42).index

# Drop half of the entries with the specified column value
df_dropped = df.drop(index=rows_to_drop_indices)

# Drop unnecessary columns
df_dropped = df_dropped.drop(columns=['State'])

# Identify and encode all categorical columns
categorical_columns = df_dropped.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df_dropped, columns=categorical_columns, drop_first=True)

# Separate features and target variable
X = df_encoded.drop(columns=['HeartDisease_Yes'])
y = df_encoded['HeartDisease_Yes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train different models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='saga'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'k-NN': KNeighborsClassifier()
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Display all relevant evaluation metrics
    print(f"{name} - Classification Report:\n", classification_rep)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)
    print("\n---------------------------------\n")