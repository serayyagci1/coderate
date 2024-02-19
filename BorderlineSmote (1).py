import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score

# Define the evaluate_model function
def evaluate_model(model, x_test, y_test, threshold=0.5):
    # Predict Test Data
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)

    # Convert predicted labels to strings if y_test is in string format
    if isinstance(y_test.iloc[0], str):
        y_pred_str = np.where(y_pred == 1, 'Yes', 'No')
    else:
        y_pred_str = y_pred.astype(str)

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred_str)
    print('Confusion Matrix:\n', cm)

    # Calculate accuracy, precision, recall, F1 Score, Cohen's Kappa Score, and area under curve (AUC)
    acc = accuracy_score(y_test, y_pred_str)
    prec = precision_score(y_test, y_pred_str, pos_label="Yes")
    rec = recall_score(y_test, y_pred_str, pos_label="Yes")
    f1 = f1_score(y_test, y_pred_str, pos_label="Yes")
    kappa = cohen_kappa_score(y_test, y_pred_str)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Print metrics
    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F1 Score:', f1)
    print('Cohen\'s Kappa Score:', kappa)
    print('Area Under Curve:', auc)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 'auc': auc, 'cm': cm}

# Drop the 'State' column because of the non-reductible input.
df = pd.read_csv("heart_2022_cleared.csv")
df.drop(['State'], axis=1, inplace=True)

# Independent Variable
X = df.drop('HeartDisease', axis=1)

# Dependent Variable
y = df['HeartDisease']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X)

# Split the data into the test and the train data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Use SMOTE for oversampling
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Use ADASYN for oversampling
adasyn = ADASYN()
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# Use Borderline SMOTE for oversampling
borderline_smote = BorderlineSMOTE()
X_train_borderline_smote, y_train_borderline_smote = borderline_smote.fit_resample(X_train, y_train)

# Scale the Data
scaler = StandardScaler()

# Scale training data for SMOTE
X_train_smote_scaled = scaler.fit_transform(X_train_smote)

# Scale training data for ADASYN
X_train_adasyn_scaled = scaler.fit_transform(X_train_adasyn)

# Scale training data for Borderline SMOTE
X_train_borderline_smote_scaled = scaler.fit_transform(X_train_borderline_smote)

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model with SMOTE
clf_R_smote = LogisticRegression(random_state=10, class_weight='balanced', max_iter=1000)
clf_R_smote.fit(X_train_smote_scaled, y_train_smote)

# Train Logistic Regression model with ADASYN
clf_R_adasyn = LogisticRegression(random_state=10, class_weight='balanced', max_iter=1000)
clf_R_adasyn.fit(X_train_adasyn_scaled, y_train_adasyn)

# Train Logistic Regression model with Borderline SMOTE
clf_R_borderline_smote = LogisticRegression(random_state=10, class_weight='balanced', max_iter=1000)
clf_R_borderline_smote.fit(X_train_borderline_smote_scaled, y_train_borderline_smote)

# Evaluate Logistic Regression model with SMOTE
print("Metrics for Logistic Regression with SMOTE")
clf_eval_smote = evaluate_model(clf_R_smote, X_test_scaled, y_test)

# Evaluate Logistic Regression model with ADASYN
print("\nMetrics for Logistic Regression with ADASYN")
clf_eval_adasyn = evaluate_model(clf_R_adasyn, X_test_scaled, y_test)

# Evaluate Logistic Regression model with Borderline SMOTE
print("\nMetrics for Logistic Regression with Borderline SMOTE")
clf_eval_borderline_smote = evaluate_model(clf_R_borderline_smote, X_test_scaled, y_test)

# Print results
print("\nMetrics of the Logistic Regression with SMOTE")
print('Accuracy:', clf_eval_smote['acc'])
print('Precision:', clf_eval_smote['prec'])
print('Recall:', clf_eval_smote['rec'])
print('F1 Score:', clf_eval_smote['f1'])
print('Cohen\'s Kappa Score:', clf_eval_smote['kappa'])
print('Area Under Curve:', clf_eval_smote['auc'])
print('Confusion Matrix:\n', clf_eval_smote['cm'])

print("\nMetrics of the Logistic Regression with ADASYN")
print('Accuracy:', clf_eval_adasyn['acc'])
print('Precision:', clf_eval_adasyn['prec'])
print('Recall:', clf_eval_adasyn['rec'])
print('F1 Score:', clf_eval_adasyn['f1'])
print('Cohen\'s Kappa Score:', clf_eval_adasyn['kappa'])
print('Area Under Curve:', clf_eval_adasyn['auc'])
print('Confusion Matrix:\n', clf_eval_adasyn['cm'])

print("\nMetrics of the Logistic Regression with Borderline SMOTE")
print('Accuracy:', clf_eval_borderline_smote['acc'])
print('Precision:', clf_eval_borderline_smote['prec'])
print('Recall:', clf_eval_borderline_smote['rec'])
print('F1 Score:', clf_eval_borderline_smote['f1'])
print('Cohen\'s Kappa Score:', clf_eval_borderline_smote['kappa'])
print('Area Under Curve:', clf_eval_borderline_smote['auc'])
print('Confusion Matrix:\n', clf_eval_borderline_smote['cm'])