
# This code cleans the Raw Dataset and makes it ML model-suitable Dataset.
import pandas as pd
# Read the Data set.
# Drop the duplicate and the NA value containing rows.
def data_clearer(file):
    df = pd.read_csv(file)
    is_null = df.isnull().values.any()
    if is_null:
        df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return (df)

# Read the csv file with the data_clearer function.
df = data_clearer("/Users/yusufberkoruc/PycharmProjects/heart_disease_predictor/heart_2022_no_nans.csv")
# Print out the Data characteristics to compare after cleaning.
print("Data Info Before Cleaning:")
df.info()
print(df.nunique())

# Merge the Heart Attack and the Angina columns to create single target variable "HeartDisease".
def disease_maker(row):
    if row['HadHeartAttack'] == "Yes" or row['HadAngina'] == "Yes":
        return "Yes"
    else:
        return "No"
df['HeartDisease'] = df.apply(disease_maker, axis=1)

#After Creating a new disease column drop the old columns.
df.drop(['HadHeartAttack', 'HadAngina',"RemovedTeeth"],axis =1,inplace =True)

# Use the clearer function to transform data into the ML model suited form.
def diabetes_clearer(element):
    dict_temp ={'No':"No", 'Yes':"Yes", 'No, pre-diabetes or borderline diabetes':"No", 'Yes, but only during pregnancy (female)':"Yes"}
    return dict_temp[element]
df["HadDiabetes"] = df['HadDiabetes'].apply(diabetes_clearer)

# Use the clearer function to transform data into the ML model suited form.
def smoker_clearer(element):
    dict_temp ={'Current smoker - now smokes every day':"Yes", 'Former smoker':"Yes", 'Never smoked':"No", 'Current smoker - now smokes some days':"Yes"}
    return dict_temp[element]
df["SmokerStatus"] = df['SmokerStatus'].apply(smoker_clearer)
# Use the clearer function to transform data into the ML model suited form.
def eCigarete_clearer(element):
    dict_temp ={'Not at all (right now)':"No", 'Use them some days':"Yes", 'Use them every day':"Yes", 'Never used e-cigarettes in my entire life':"No"}
    return dict_temp[element]
df["ECigaretteUsage"] = df['ECigaretteUsage'].apply(eCigarete_clearer)

# Use the clearer function to transform data into the ML model suited form.
def RaceEthnicityCategory_clearer(element):
    dict_temp ={'Hispanic':"Hispanic", 'Other race only, Non-Hispanic':"Other", 'Multiracial, Non-Hispanic':"Multiracial", 'White only, Non-Hispanic':"White", 'Black only, Non-Hispanic':"Black"}
    return dict_temp[element]
df["RaceEthnicityCategory"] = df['RaceEthnicityCategory'].apply(RaceEthnicityCategory_clearer)

# Use the clearer function to transform data into the ML model suited form.
def TetanusLast10Tdap_clearer(element):
   return  element.split(",")[0]
df["TetanusLast10Tdap"] = df['TetanusLast10Tdap'].apply(TetanusLast10Tdap_clearer)

# Use the clearer function to transform data into the ML model suited form.
def CovidPos_clearer(element):
    dict_temp = {'No':"No", 'Tested positive using home test without a health professional':"Yes", 'Yes':"Yes"}
    return dict_temp[element]
df["CovidPos"] = df['CovidPos'].apply(CovidPos_clearer)


# Drop the redundant columns which are captured in the BMI column.
df.drop(['HeightInMeters', 'WeightInKilograms'],axis =1,inplace =True)

#Save the Cleaned Data.
df.to_csv("heart_2022_cleared.csv",index=False)

#Check the features of the Clean Data.
print("Data Info After Cleaning")
df.info()
print(df.nunique())
for i in df.columns:
    print(df[i].value_counts())


