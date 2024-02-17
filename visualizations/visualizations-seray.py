import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
# Load the data from the CSV file
df = pd.read_csv("heart-data.csv")

# Define BMI categories based on thresholds
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi <= 24.9:
        return 'Normal Weight'
    elif 25 <= bmi <= 30:
        return 'Overweight'
    else:
        return 'Obese'

# Apply the categorization to create a new column 'BMICategory'
df['BMICategory'] = df['BMI'].apply(categorize_bmi)

# Remove leading and trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Create a dataframe for percentage calculation
percentage_df = df.groupby(['BMICategory', 'HadHeartAttack']).size().unstack().fillna(0)
percentage_df['HadHeartAttack_original'] = percentage_df['Yes']

# Adding total column
percentage_df['Total'] = percentage_df.sum(axis=1)

# Calculate the percentage 
percentage_df['HeartDiseasePercentage'] = (percentage_df['Yes'] / percentage_df['Total']) * 100
# Define the order of BMI categories
bmi_order = ['Underweight', 'Normal Weight', 'Overweight', 'Obese']

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='BMICategory', y='HeartDiseasePercentage', data=percentage_df.reset_index())
plt.title('Heart Disease Percentage by BMI Category')
plt.xlabel('BMI Category')
plt.ylabel('Heart Disease Percentage')
plt.show()

# COVID plot

# Remove leading and trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Filter dataframe for 'CovidPos' with values 'Yes' and 'No'
df = df[df['CovidPos'].isin(['Yes', 'No'])]

# Create a dataframe for percentage calculation
percentage_df = df.groupby(['CovidPos', 'HadHeartAttack']).size().unstack().fillna(0)

# Add total column
percentage_df['Total'] = percentage_df.sum(axis=1)

# Calculate the percentage 
percentage_df['HadHeartAttackPercentage'] = (percentage_df['Yes'] / percentage_df['Total']) * 100

# Reset the index before plotting
percentage_df.reset_index(inplace=True)

# Print available columns for debugging
print(percentage_df.columns)

# Plotting with matplotlib barplot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting bars
for i, (index, row) in enumerate(percentage_df.iterrows()):
    ax.bar(x=row['CovidPos'], height=row['HadHeartAttackPercentage'], color='C0' if row['CovidPos'] == 'Yes' else 'C1', label=row['CovidPos'] if i == 0 else "", alpha=0.7)

# Customize the plot
ax.set_title('Heart Disease Percentage By Covid Infection Category')
ax.set_xlabel('Had/Has Covid')
ax.set_ylabel('Heart Disease Percentage')

plt.show()


# Age Violin plot


# Extract numerical values from 'AgeCategory' and convert to integers
df['AgeCategoryNumeric'] = df['AgeCategory'].str.extract('(\d+)').astype(float)

# Order the dataframe by the numerical values in descending order
df_sorted = df.sort_values(by='AgeCategoryNumeric', ascending=False)

# Define the order for 'HadHeartAttack' variable
order = ['Yes', 'No']

# Age Violin plot with ordered categories
plt.figure(figsize=(10, 6))
sns.violinplot(x='HadHeartAttack', y='AgeCategory', data=df_sorted, order=order)
plt.title('Effect of Age on Heart Disease')
plt.xlabel('Heart Disease')
plt.ylabel('Age Category')
plt.show()



#Depression

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})
df['HadDepressiveDisorder'] = df['HadDepressiveDisorder'].map({'Yes': 1, 'No': 0})

# Create a cross-tabulation
ct = pd.crosstab(df['HadDepressiveDisorder'], df['HadHeartAttack'], margins=True, margins_name='Total')

# Calculate percentages
percentage_df = ct.div(ct['Total'], axis=0) * 100

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x='HadDepressiveDisorder', y=1, data=percentage_df[:-1])  # '1' represents 'Yes' for heart disease
plt.title('Percentage of Heart Disease in each group of Depression')
plt.xlabel('Depression')
plt.ylabel('Percentage of Heart Disease')

# Update x-axis labels
plt.xticks([0, 1], ['No', 'Yes'])

plt.show()


#Diabetes

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Keep only 'Yes' and 'No' in the 'HadDiabetes' column
df = df[df['HadDiabetes'].isin(['Yes', 'No'])]

# Create a binary column for diabetes
df['DiabetesBinary'] = df['HadDiabetes'].map({'Yes': 1, 'No': 0})

# Create a cross-tabulation
ct = pd.crosstab(df['DiabetesBinary'], df['HadHeartAttack'], margins=True, margins_name='Total')

# Calculate percentages
percentage_df = ct.div(ct['Total'], axis=0) * 100

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x='DiabetesBinary', y=1, data=percentage_df[:-1])  # '1' represents 'Yes' for heart disease
plt.title('Percentage of Heart Disease in each group of Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Percentage of Heart Disease')

# Update x-axis labels
plt.xticks([0, 1], ['No', 'Yes'])

plt.show()


#Cigarettes


#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Create a cross-tabulation
ct = pd.crosstab(df['SmokerStatus'], df['HadHeartAttack'], margins=True, margins_name='Total')

# Calculate percentages
percentage_df = ct.div(ct['Total'], axis=0) * 100

# Reset the index
percentage_df = percentage_df.reset_index()

# Exclude the "Total" category
percentage_df = percentage_df[percentage_df['SmokerStatus'] != 'Total']

# Wrap long labels
max_label_width = 12  
percentage_df['SmokerStatus'] = percentage_df['SmokerStatus'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=max_label_width)))

# Order values on the x-axis
order = percentage_df.sort_values(by=1)['SmokerStatus']

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='SmokerStatus', y=1, data=percentage_df, order=order)  # '1' represents 'Yes' for heart disease
plt.title('Percentage of Heart Disease for Different Smoker Status')
plt.xlabel('Smoker Status')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

plt.tight_layout() 
plt.show()

#Ecigaretteusage

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Create a cross-tabulation
ct = pd.crosstab(df['ECigaretteUsage'], df['HadHeartAttack'], margins=True, margins_name='Total')

# Calculate percentages
percentage_df = ct.div(ct['Total'], axis=0) * 100

# Reset the index
percentage_df = percentage_df.reset_index()

# Exclude the "Total" category
percentage_df = percentage_df[percentage_df['ECigaretteUsage'] != 'Total']

# Wrap long labels
max_label_width = 12  # Set the maximum width for the labels
percentage_df['ECigaretteUsage'] = percentage_df['ECigaretteUsage'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=max_label_width)))

# Order values on the x-axis
order = percentage_df.sort_values(by=1)['ECigaretteUsage']

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='ECigaretteUsage', y=1, data=percentage_df, order=order)  # '1' represents 'Yes' for heart disease
plt.title('Percentage of Heart Disease for each E-cigarette Usage Status')
plt.xlabel('E-cigarette Usage Status')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

plt.tight_layout()  # Ensure the layout is tight
plt.show()


#Sleephours

# Convert sleep hours to numeric type 
df['SleepHours'] = pd.to_numeric(df['SleepHours'], errors='coerce')

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Create a cross-tabulation
ct = pd.crosstab(df['SleepHours'], df['HadHeartAttack'], margins=True, margins_name='Total')

# Calculate percentages
percentage_df = ct.div(ct['Total'], axis=0) * 100

# Reset the index
percentage_df = percentage_df.reset_index()

# Exclude the "Total" category
percentage_df = percentage_df[percentage_df['SleepHours'] != 'Total']

# Order values on the x-axis based on sleep hours
order = percentage_df.sort_values(by='SleepHours')['SleepHours']

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='SleepHours', y=1, data=percentage_df, order=order)  # '1' represents 'Yes' for heart disease
plt.title('Percentage of Heart Disease for each Sleep Time')
plt.xlabel('Sleep Hours')
plt.ylabel('Percentage of Heart Disease')

# Display x-axis as integers
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Ensure the layout is tight
plt.show()


#Sex

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Calculate percentages
percentage_df = df.groupby('Sex')['HadHeartAttack'].value_counts(normalize=True).unstack() * 100

# Plotting
plt.figure(figsize=(8, 6))
percentage_df.plot(kind='bar', stacked=True, color=['C1', 'C0'], width=0.8)
plt.title('Percentage of Heart Disease by Sex')
plt.xlabel('Sex')
plt.ylabel('Percentage Of Heart Disease')
plt.legend(title='HadHeartAttack', loc='upper right', labels=['No', 'Yes'])
plt.show()

#Physical Activity

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Calculate percentages
percentage_df = df.groupby('PhysicalActivities')['HadHeartAttack'].value_counts(normalize=True).unstack() * 100

# Plotting
plt.figure(figsize=(8, 6))
percentage_df.plot(kind='bar', stacked=True, color=['C1', 'C0'], width=0.8)
plt.title('Percentage of Heart Disease by Physical Activity')
plt.xlabel('Physical Activity')
plt.ylabel('Percentage of Heart Disease')
plt.legend(title='HadHeartAttack', loc='upper right', labels=['No', 'Yes'])
plt.show()


#Alcohol

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Calculate percentages
percentage_df = df.groupby('AlcoholDrinkers')['HadHeartAttack'].value_counts(normalize=True).unstack() * 100

# Plotting
plt.figure(figsize=(8, 6))
percentage_df.plot(kind='bar', stacked=True, color=['C1', 'C0'], width=0.8)
plt.title('Percentage of Heart Disease by Alcohol Consumption')
plt.xlabel('Alcohol Drinkers')
plt.ylabel('Percentage')
plt.legend(title='HadHeartAttack', loc='upper right', labels=['No', 'Yes'])
plt.show()



#COPD

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Calculate percentages
percentage_df = df.groupby('HadCOPD')['HadHeartAttack'].value_counts(normalize=True).unstack() * 100

# Plotting
plt.figure(figsize=(8, 6))
percentage_df.plot(kind='bar', stacked=True, color=['C1', 'C0'], width=0.8)
plt.title('Percentage of Heart Disease by COPD')
plt.xlabel('COPD')
plt.ylabel('Percentage')
plt.legend(title='Had COPD', loc='upper right', labels=['No', 'Yes'])
plt.show()


#HIV

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Calculate percentages
percentage_df = df.groupby('HIVTesting')['HadHeartAttack'].value_counts(normalize=True).unstack() * 100

# Plotting
plt.figure(figsize=(8, 6))
percentage_df.plot(kind='bar', stacked=True, color=['C1', 'C0'], width=0.8)
plt.title('Percentage of Heart Disease by HIV')
plt.xlabel('HIV')
plt.ylabel('Percentage')
plt.legend(title='Has HIV', loc='upper right', labels=['No', 'Yes'])
plt.show()


#Chest scan

#Convert string yes/no values to boolean
df['HadHeartAttack'] = df['HadHeartAttack'].map({'Yes': 1, 'No': 0})

# Calculate percentages
percentage_df = df.groupby('ChestScan')['HadHeartAttack'].value_counts(normalize=True).unstack() * 100

# Plotting
plt.figure(figsize=(8, 6))
percentage_df.plot(kind='bar', stacked=True, color=['C1', 'C0'], width=0.8)
plt.title('Percentage of Heart Disease by Chest Scan')
plt.xlabel('Chest Scan')
plt.ylabel('Percentage')
plt.legend(title='Chest Scan', loc='upper right', labels=['No', 'Yes'])
plt.show()
