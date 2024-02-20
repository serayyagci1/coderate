import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import joblib

#Transform input_dict into the data frame that is model predictable.
def input_taker(input_dict,column_list_num,column_list_bool):
    output_dict = dict()
    #Directly put the numerical value into the output_dict
    for i in column_list_num:
        if i in input_dict:
            output_dict[i] = input_dict[i]
    #If the key == value then set it true otherwise false
    for j in column_list_bool:
        key,value = j.split("_",1)
        if input_dict[key][0] == value:
            output_dict[j] = True
        else:
            output_dict[j] = False
    data_frame = pd.DataFrame.from_dict(output_dict)
    return data_frame

#Predict the single row of the data
def single_predictor(model,scaler,series):
    scaled_series = scaler.transform(series)
    prediction_categorical = model.predict(scaled_series)
    prediction_prob = model.predict_proba(scaled_series)
    return prediction_categorical,prediction_prob

#Evaluate the Model Metrics
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
    y_pred_proba = model.predict_proba(x_test)[:,1]

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba, pos_label="Yes")
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm, 'y_pred_prob':y_pred_proba}


#Read the final clean data set suitable for the ML model.
result_df = pd.read_csv("/Users/yusufberkoruc/PycharmProjects/heart_disease_predictor/final_evaluation_data.csv")

#Independent Variable
X = result_df.drop('HeartDisease', axis= 1)

#Dependent Variables
y = result_df['HeartDisease']

#Split the data into the test and the train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Use the one hot encoding for the categorical data
X_train_encoded = pd.get_dummies(X_train)

X_test_encoded = pd.get_dummies(X_test)


# Scale the training data
scaler_train = StandardScaler().fit(X_train_encoded)
X_train_encoded_scaled = scaler_train.transform(X_train_encoded)

# Save the scaler
joblib.dump(scaler_train, 'scaler_train.save')


# Scale test data
scaler_test = StandardScaler().fit(X_test_encoded)
X_test_encoded_scaled = scaler_test.transform(X_test_encoded)
# Save the scaler
joblib.dump(scaler_test, 'scaler_test.save')



# Train the model with Logistic Regression
clf_R = LogisticRegression(random_state=0)
clf_R.fit(X_train_encoded_scaled, y_train)
# Save the model
joblib.dump(clf_R, 'model_R.save')

# Train the model with Tree Classifier
clf_T = tree.DecisionTreeClassifier(random_state=0)
clf_T.fit(X_train_encoded_scaled, y_train)
# Save the model
joblib.dump(clf_T, 'model_T.save')

# Train the model with the KNeighborsClassifier
clf_K = KNeighborsClassifier(n_neighbors = 5)
clf_K.fit(X_train_encoded_scaled, y_train)

# Save the model
joblib.dump(clf_K, 'model_K.save')


# Evaluate Model Logistic Regression
clf_eval_L = evaluate_model(clf_R, X_test_encoded_scaled, y_test)

# Print result
print("The Metrics of the LogisticRegression")
print('Accuracy:', clf_eval_L['acc'])
print('Precision:', clf_eval_L['prec'])
print('Recall:', clf_eval_L['rec'])
print('F1 Score:', clf_eval_L['f1'])
print('Cohens Kappa Score:', clf_eval_L['kappa'])
print('Area Under Curve:', clf_eval_L['auc'])
print("\n")


# Evaluate Model Decision Tree
clf_eval_T = evaluate_model(clf_T, X_test_encoded_scaled, y_test)

# Print result
print("The Metrics of the DecisionTree")
print('Accuracy:', clf_eval_T['acc'])
print('Precision:', clf_eval_T['prec'])
print('Recall:', clf_eval_T['rec'])
print('F1 Score:', clf_eval_T['f1'])
print('Cohens Kappa Score:', clf_eval_T['kappa'])
print('Area Under Curve:', clf_eval_T['auc'])
print("\n")



# Evaluate Model KneighborsClassifier
clf_eval_K = evaluate_model(clf_K, X_test_encoded_scaled, y_test)

# Print result
print("The Metrics of the KNeighboursClassifier")
print('Accuracy:', clf_eval_K['acc'])
print('Precision:', clf_eval_K['prec'])
print('Recall:', clf_eval_K['rec'])
print('F1 Score:', clf_eval_K['f1'])
print('Cohens Kappa Score:', clf_eval_K['kappa'])
print('Area Under Curve:', clf_eval_K['auc'])
print("\n")

#Recall the best model and the scaler required for prediction
model = joblib.load('model_R.save')
scaler_test = joblib.load('scaler_test.save')
scaler_train = joblib.load("scaler_train.save")




# Check whether the single predictor works as same as the bulk prediction.
# Predict the test data.
"""prediction = model.predict(X_test_encoded_scaled)
for i in range(len(X_test_encoded)):
    prediction_categorical = single_predictor(model,scaler_test,X_test_encoded.iloc[[i]])[0]
    if (prediction_categorical == prediction[i]):
        Flag = True
    else:
        print("life is hard")
        break
if Flag:
    print("The single_predictor worked properly")"""


#One hot encoded list of the column names(numerical and boolean)
column_list_numerical =['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
column_list_boolean  = [ 'Smoking_No','Smoking_Yes', 'AlcoholDrinking_No', 'AlcoholDrinking_Yes', 'Stroke_No',
       'Stroke_Yes', 'DiffWalking_No', 'DiffWalking_Yes', 'Sex_Female',
       'Sex_Male', 'AgeCategory_18-24', 'AgeCategory_25-29',
       'AgeCategory_30-34', 'AgeCategory_35-39', 'AgeCategory_40-44',
       'AgeCategory_45-49', 'AgeCategory_50-54', 'AgeCategory_55-59',
       'AgeCategory_60-64', 'AgeCategory_65-69', 'AgeCategory_70-74',
       'AgeCategory_75-79', 'AgeCategory_80 or older',
       'Race_American Indian/Alaskan Native', 'Race_Asian', 'Race_Black',
       'Race_Hispanic', 'Race_Multiracial', 'Race_Other', 'Race_White',
       'Diabetic_No', 'Diabetic_Yes', 'PhysicalActivity_No',
       'PhysicalActivity_Yes', 'GenHealth_Excellent', 'GenHealth_Fair',
       'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good', 'Asthma_No',
       'Asthma_Yes', 'KidneyDisease_No', 'KidneyDisease_Yes', 'SkinCancer_No',
       'SkinCancer_Yes']


#Example Usage
random_int =  np.random.randint(0,len(X_test))
random_input_dict = X_test.iloc[[random_int]].to_dict(orient = 'list')

#Predict random Value with the Model.
random_input_dict ={'BMI': [20], 'Smoking': ['Yes'], 'AlcoholDrinking': ['Yes'],
                    'Stroke': ['Yes'], 'PhysicalHealth': [0.0], 'MentalHealth': [0.0], 'DiffWalking': ['Yes'],
                    'Sex': ['Male'], 'AgeCategory': ['70-74'], 'Race': ['White'], 'Diabetic': ['Yes'],
                    'PhysicalActivity': ['No'], 'GenHealth': ['Excellent'], 'SleepTime': [8.0], 'Asthma': ['No'],
                    'KidneyDisease': ['No'], 'SkinCancer': ['No']}
random_series = input_taker(random_input_dict,column_list_numerical,column_list_boolean)
prediction_categorical,prediction_probability = single_predictor(model,scaler_train,random_series)

#Display the Prediction Results.
dict_2 = {"Yes":"Positive","No":"Negative"}
dict_3 = {"Yes":prediction_probability[0][1],"No":prediction_probability[0][0]}
print(f" Ml model predicts the Heart disease as {dict_2[prediction_categorical[0]]} with the probability of {dict_3[prediction_categorical[0]]} ")


# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
clf_T_score = [clf_eval_T['acc'], clf_eval_T['prec'], clf_eval_T['rec'], clf_eval_T['f1'], clf_eval_T['kappa']]
clf_K_score = [clf_eval_K['acc'], clf_eval_K['prec'], clf_eval_K['rec'], clf_eval_K['f1'], clf_eval_K['kappa']]
clf_R_score = [clf_eval_L['acc'], clf_eval_L['prec'], clf_eval_L['rec'], clf_eval_L['f1'], clf_eval_L['kappa']]


## Set position of bar on X axis
r1 = np.arange(len(clf_T_score))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

## Make the plot
ax1.bar(r1, clf_T_score, width=barWidth, edgecolor='white', label='Decision Tree')
ax1.bar(r2, clf_K_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')
ax1.bar(r3,clf_R_score,width=barWidth, edgecolor='white', label='Logistic Regression')

## Configure x and y axis
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.25) for r in range(len(clf_T_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(clf_eval_T['fpr'], clf_eval_T['tpr'], label='Decision Tree, auc = {:0.5f}'.format(clf_eval_T['auc']))
ax2.plot(clf_eval_K['fpr'], clf_eval_K['tpr'], label='K-Nearest Neighbor, auc = {:0.5f}'.format(clf_eval_K['auc']))
ax2.plot(clf_eval_L['fpr'], clf_eval_L['tpr'], label='Logistic Regression, auc = {:0.5f}'.format(clf_eval_L['auc']))

## Configure x and y axis
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')

## Create legend & title
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)


# Intitialize figure with two plots
fig, axs = plt.subplots(1, 3, figsize=(10, 10))

fig.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')


sns.heatmap(clf_eval_L["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax = axs[0])
axs[0].set_title("Logistic Regression")
axs[0].set_xlabel("Prediction")
axs[0].set_ylabel("Real Value")

sns.heatmap(clf_eval_K["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax = axs[1])
axs[1].set_title("Decision Tree")
axs[1].set_xlabel("Prediction")
axs[1].set_ylabel("Real Value")

sns.heatmap(clf_eval_T["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax = axs[2])
axs[2].set_title("KNeighbor Classifier")
axs[2].set_xlabel("Prediction")
axs[2].set_ylabel("Real Value")

plt.tight_layout()
plt.show()
