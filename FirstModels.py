import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree

#Drop the state column because of the non-reductible input.
df = pd.read_csv("heart_2022_cleared.csv")
df.drop(['State'],axis =1,inplace =True)

#Independent Variable
X = df.drop('HeartDisease', axis= 1)

#Dependent Variables
y = df['HeartDisease']

#Split the data into the test and the train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Print the shape of the splitted data
"""print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of testing label:', y_test.shape)
"""
#Use the one hot encoding for the categorical data
X_train_encoded = pd.get_dummies(X_train)

X_test_encoded = pd.get_dummies(X_test)

#Print the shape of the Encoded data
"""print('Shape of training feature encoded:', X_train_encoded.shape)
print('Shape of testing feature encoded:', X_test_encoded.shape)"""

#Scale the Data
scaler = StandardScaler()

# Scale training data
X_train_encoded = scaler.fit_transform(X_train_encoded)

# Scale test data
X_test_encoded = scaler.fit_transform(X_test_encoded)

#Print the shape of the Encoded and scaled data

"""print('Shape of training feature encoded:', X_train_encoded.shape)
print('Shape of testing feature encoded:', X_test_encoded.shape)"""


# Train the model with different classification algorithms
clf_K = KNeighborsClassifier(n_neighbors = 5)
clf_K.fit(X_train_encoded, y_train)

clf_R = LogisticRegression(random_state=0)
clf_R.fit(X_train_encoded, y_train)

clf_T = tree.DecisionTreeClassifier(random_state=0)
clf_T.fit(X_train_encoded, y_train)

def evaluate_model(model, x_test, y_test):
    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, pos_label="Yes")
    rec = metrics.recall_score(y_test, y_pred, pos_label="Yes")
    f1 = metrics.f1_score(y_test, y_pred, pos_label="Yes")
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba, pos_label="Yes")
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm, 'y_pred_prob':y_pred_proba}


# Evaluate Model KNeighborsClassifier
knn_eval = evaluate_model(clf_K, X_test_encoded, y_test)

# Print result
print("The Metrics of the KNeighborsClassifier")
print('Accuracy:', knn_eval['acc'])
print('Precision:', knn_eval['prec'])
print('Recall:', knn_eval['rec'])
print('F1 Score:', knn_eval['f1'])
print('Cohens Kappa Score:', knn_eval['kappa'])
print('Area Under Curve:', knn_eval['auc'])
print('Confusion Matrix:\n', knn_eval['cm'])


# Evaluate Model Logistic Regression
clf_eval = evaluate_model(clf_R, X_test_encoded, y_test)

# Print result
print("The Metrics of the LogisticRegression")
print('Accuracy:', clf_eval['acc'])
print('Precision:', clf_eval['prec'])
print('Recall:', clf_eval['rec'])
print('F1 Score:', clf_eval['f1'])
print('Cohens Kappa Score:', clf_eval['kappa'])
print('Area Under Curve:', clf_eval['auc'])
print('Confusion Matrix:\n', clf_eval['cm'])

# Evaluate Model
clf_eval = evaluate_model(clf_T, X_test_encoded, y_test)

#Print result
print("The Metrics of the DecisionTree")
print('Accuracy:', clf_eval['acc'])
print('Precision:', clf_eval['prec'])
print('Recall:', clf_eval['rec'])
print('F1 Score:', clf_eval['f1'])
print('Cohens Kappa Score:', clf_eval['kappa'])
print('Area Under Curve:', clf_eval['auc'])
print('Confusion Matrix:\n', clf_eval['cm'])


random_int =  np.random.randint(0,len(X_test))
random_row = [X_test_encoded[random_int]]
y_pred_categorical = clf_R.predict(random_row)
y_pred = clf_R.predict_proba(random_row)[::,1]
print(y_pred_categorical)