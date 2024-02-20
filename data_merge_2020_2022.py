#This code cleans the 2.nd Raw Dataset and makes it ML model-suitable Dataset.
#This code merges two Datasets and samples the final Dataset to create a balanced dataset for the ML training.
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
# Standardize the age values between the datasets.
def age_closer(element):
    if element.startswith("Age"):
        if "or" in element:
            element = '80 or older'
        else:
            element = "-".join(element.split(" ")[1:4:2])
    return element
# Use the clearer function to transform data into the ML model suited form.
def diabetes_clearer(element):
    dict_temp ={'No':"No", 'Yes':"Yes", 'No, borderline diabetes':"No", 'Yes (during pregnancy)':"Yes"}
    return dict_temp[element]


# Read the csv files with the data_clearer function.
df_2020 = data_clearer("/Users/yusufberkoruc/PycharmProjects/heart_disease_predictor/heart_2020_cleaned.csv")
df_2022 = pd.read_csv("/Users/yusufberkoruc/PycharmProjects/heart_disease_predictor/heart_2022_cleared.csv")

#Apply the diabetes_clearer function.
df_2020["Diabetic"] = df_2020['Diabetic'].apply(diabetes_clearer)

# Fİnd the column names to standardize column names between two datasets.
list_2020_columns = list(df_2020.columns)
list_2022_columns = list(df_2022.columns)

# Standardize the column names.
rename_dict = dict()
for i in list_2020_columns:
    for k in list_2022_columns:
        if i in k:
            rename_dict[k] = i
#Final Dictionary for the column name standardization.
dict_rename = {'HadDiabetes':'Diabetic','PhysicalActivities':'PhysicalActivity','SleepHours':'SleepTime','DifficultyWalking':'DiffWalking','GeneralHealth':'GenHealth','AlcoholDrinkers':'AlcoholDrinking','SmokerStatus':'Smoking','HeartDisease': 'HeartDisease', 'BMI': 'BMI', 'HadStroke': 'Stroke', 'PhysicalHealthDays': 'PhysicalHealth', 'MentalHealthDays': 'MentalHealth', 'Sex': 'Sex', 'AgeCategory': 'AgeCategory', 'RaceEthnicityCategory': 'Race', 'HadAsthma': 'Asthma', 'HadKidneyDisease': 'KidneyDisease', 'HadSkinCancer': 'SkinCancer'}

#Change the column names.
df_2022 = df_2022.rename(columns=dict_rename)

#Subset the 2022 data to have same columns.
df_2022 = df_2022[dict_rename.values()]

#Merge the Datasets.
concat_df = pd.concat([df_2020,df_2022], ignore_index=True)


# Apply the age_closer function
concat_df["AgeCategory"] = concat_df['AgeCategory'].apply(age_closer)

# Split the Dataset as the positive and negative.
concat_df_positive = concat_df[concat_df['HeartDisease'] == "Yes" ]
concat_df_negative = concat_df[concat_df['HeartDisease'] == "No" ]

# Sample the negative dataset to evade bias.
sampled_df = concat_df_negative.sample(len(concat_df_positive),random_state=42)

#Merge the sampled negative dataset and the positive data set.
result_df = pd.concat([sampled_df , concat_df_positive], ignore_index=True)

#Save the final dataset to be used for the ML training.
result_df.to_csv("final_evaluation_data.csv",index=False)

#Check the features of the Final Dataset.

result_df.info()
print(result_df.nunique())
for i in result_df.columns:
    print(result_df[i].value_counts())

