# An Interactive Exploration of Heart Disease Indicators

## Description
According to the CDC, heart disease is a leading cause of death for people of most races in the U.S. About half of all Americans (47%) have at least one major risk factor for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicators include diabetes status, obesity (high BMI), level of physical activity, or alcohol consumption. The application of machine learning methods to detect patterns in the data can predict a patient's susceptibility to heart disease. This tool helps them to make informed decisions about lifestyle modifications to potentially reduce their risk of heart disease.

The [dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data) used for this project was taken from Kaggle.




## Functionalities
The project has the following functionalities, which makes use of external libraries to enhance our platform's capabilities:

### Data Sources and Retrieval

The data for our project is sourced from a comprehensive dataset available on Kaggle, originally collected by the CDC.
The final data set  contains 246014 instances (after data pre-processing). 36 relevant features have been extracted from a maximum of 40 in the total data.

The Features include (with description) :
1. **State**: The state where the individual resides.
2. **Sex**: Gender of the individual (Male/Female).
3. **GeneralHealth**: Overall health status reported by the individual (e.g., Very good, Good, Fair).
4. **PhysicalHealthDays**: Number of days the individual experienced physical health issues.
5. **MentalHealthDays**: Number of days the individual experienced mental health issues.
6. **LastCheckupTime**: Time since the individual's last medical checkup.
7. **PhysicalActivities**: Whether the individual engages in physical activities (Yes/No).
8. **SleepHours**: Number of hours of sleep the individual gets.
9. **HadStroke**: Whether the individual had a stroke (Yes/No).
10. **HadAsthma**: Whether the individual had asthma (Yes/No).
11. **HadSkinCancer**: Whether the individual had skin cancer (Yes/No).
12. **HadCOPD**: Whether the individual had Chronic Obstructive Pulmonary Disease (COPD) (Yes/No).
13. **HadDepressiveDisorder**: Whether the individual had a depressive disorder (Yes/No).
14. **HadKidneyDisease**: Whether the individual had kidney disease (Yes/No).
15. **HadArthritis**: Whether the individual had arthritis (Yes/No).
16. **HadDiabetes**: Whether the individual had diabetes (Yes/No).
17. **DeafOrHardOfHearing**: Whether the individual is deaf or hard of hearing (Yes/No).
18. **BlindOrVisionDifficulty**: Whether the individual has blindness or vision difficulty (Yes/No).
19. **DifficultyConcentrating**: Whether the individual experiences difficulty concentrating (Yes/No).
20. **DifficultyWalking**: Whether the individual experiences difficulty walking (Yes/No).
21. **DifficultyDressingBathing**: Whether the individual experiences difficulty dressing or bathing (Yes/No).
22. **DifficultyErrands**: Whether the individual experiences difficulty running errands (Yes/No).
23. **SmokerStatus**: Smoking status of the individual (Yes/No).
24. **ECigaretteUsage**: Whether the individual uses e-cigarettes (Yes/No).
25. **ChestScan**: Whether the individual had a chest scan (Yes/No).
26. **RaceEthnicityCategory**: Race or ethnicity category of the individual.
27. **AgeCategory**: Age category of the individual.
28. **BMI**: Body Mass Index (BMI) of the individual.
29. **AlcoholDrinkers**: Whether the individual drinks alcohol (Yes/No).
30. **HIVTesting**: Whether the individual has undergone HIV testing (Yes/No).
31. **FluVaxLast12**: Whether the individual received a flu vaccine in the last 12 months (Yes/No).
32. **PneumoVaxEver**: Whether the individual has ever received a pneumococcal vaccine (Yes/No).
33. **TetanusLast10Tdap**: Whether the individual received a tetanus vaccine in the last 10 years (Yes/No).
34. **HighRiskLastYear**: Whether the individual was at high risk for any disease in the last year (Yes/No).
35. **CovidPos**: Whether the individual tested positive for COVID-19 (Yes/No).
36. **HeartDisease**: The target variable indicating whether the individual has heart disease (Yes/No).

### Data Storage and Handling

The dataset was imported as a dataset using Pandas as a CSV file in Python, which allows data manipulation and analysis. This approach ensures compatibility with various data analysis tools without needing a separate database system.

### Data Pre-processing
A preliminary data exploration was performed on the dataset  to understand its structure, characteristics, and distribution of variables. The final dataset was obtained by merging the data from 2020 and 2022 to address data imbalance and improve the model accuracy. The data was cleaned by dropping missing values(NA), duplicates and standardizing column names and values (For example: Age Categories). The 2022 dataset was subsetted and merged with the 2020 dataset as a common data frame ('**final_evalulation_data.csv**').

### ML Model Implementation
- The final clean dataset is split into training and test datasets. Feature encoding (One-hot encoding) and Feature scaling are applied separately to both datasets.
- Different approaches to handle imbalanced datasets were implemented. These include Logistic regression with SMOTE, ADASYN,  and combined method for oversampling
- Logistic regression and Decision Tree classifier models are chosen for this dataset. For each trained model, evaluation metrics:  accuracy, precision, recall, F1-score, Cohen's Kappa score, and area under the ROC curve (AUC) are calculated using the **evaluate_model()** function. Confusion matrices are also computed for both models to assess performance.
- The trained models, along with the scalers used for feature scaling, are saved to disk using **joblib.dump()** for future use.
- The prediction result (whether the prediction is positive or negative for heart disease) along with the probability is displayed.
   
### Statistical Analysis
The following Statistical analyses were performed in the dataset:
- Correlation Matrix heatmap:  To visualize the correlation between all the features in the dataset.
- Chi-Square Test for Categorical Variables: chi-square analysis for all pairs of categorical variables in the dataset. A low p-value(p<0.05) suggests  dependence between variables. The test is performed using **chi2_contingency()** from scipy.stats
- Outlier Analysis for Numerical Columns: For each numerical column in the dataset, outliers were visualized using Seaborn **sns.boxplot()**. Outliers were detected using the interquartile range (IQR) method.
- T-Test for Comparing Groups: t-test to compare two groups based on a categorical variable (e.g., smoking status or gender) for the occurrence of heart disease(Yes or No).  The t-test uses **ttest_ind()** from scipy.stats.

### Interface
A web application for heart disease prediction and visualization was created using Django called **heartProjectApp**. The application has the following functionalities: 

- Ajax calls for dynamic interaction, categorization, and storage of visualization images in SQLite database. The images can be retrieved on selection by the user.
- Interactive form with Django's form function for users to input their data related to heart disease risk factors, such as Age, BMI, Race, etc., and used the evaluation script(**model_r**) to evaluate user data with the ML model, and reflect user result. (Heart disease Positive/Negative with probability)
- A choropleth map of US states' heart disease percentages. This visualization is based on our data combined with a US GeoJSON file containing the geographical boundaries of states.

### Visualizations
Several plots have been created to visualize the data for understanding the distribution of different variables. These include: 

1. Bar plot showing the percentage of heart disease cases in different BMI categories.
2. Violin plot demonstrating the distribution of ages for individuals with and without heart disease.
3. Various Bar plots illustrating the percentage of heart disease cases among different categorical data(e.g. Diabetes, Depressive disorders, Smoking Status, COPD etc.)
4. Pie chart illustrating the distribution of heart disease cases among different racial and ethnic categories.
5. Strip plot demonstrating the distribution of BMI values for individuals with and without heart disease.
6. A heatmap illustrating the correlation matrix among selected numeric columns, including heart disease and other variables.
7. Two histograms showing the distribution of BMI values among individuals with and without heart disease.

### Installation and Usage
To run the web application, The user needs to access the project files from GitHub or by direct download. Go to the project file directory using the Terminal in your Python environment and install the required dependencies (listed in the **requirements.txt** file) such as pip, Django, SQLite, etc.
Following this , :
- Run pip install -r requirements.txt to install dependencies. The program uses SQLite as default database. You have to modify the database settings in **settings.py** file if you want to use other databases like MySQL or PostgreSQL.
  
To make Django create necessary database tables and structures :
- Run python manage.py makemigrations
- Run python manage.py migrate
  
And Finally , to run the server and open your application locally : 
- Run python manage.py runserver
- Navigate to [http://127.0.0.1:8000/heartProjectApp/](http://127.0.0.1:8000/heartProjectApp/) to access the application





## Timeline
![The following is our timeline for the project](https://github.com/serayyagci1/coderate/blob/master/project_timeline.png?raw=true)

## Dependencies
The libraries used in this project include: 
- For Data Handling and Pre-processing: Pandas, NumPy
- For ML model implementation : Pandas(pd) , NumPy(np) , Scikit-learn (sklearn) , joblib
- For Data Visualization : Matplotlib(plt) , Seaborn(sns) and Plotly Express
- For Statistical Analysis: Pandas, NumPy, SciPy(scipy.stats)
- For Web interface: Django(django.db.models, django. forms, json ,django.shortcuts.render), pip , Matplotlib , Plotly Express, and Seaborn
  

## Group Details
- Group name: Coderate
- Group code: G04
- Tutor : Sven Sören Lange
- Group leader: Seray Yağcı
- Group members: Seray Yağcı , Yusuf Berk Oruç , Prashanth Sridhar ,  Berfin Taşkın , Emre Semercioğlu

The members contributed to the project in the following ways: 

- Data Gathering : Yusuf , Prashanth , Seray , Berfin and Emre
- Data Cleaning and Handling: Yusuf
- ML Model : Yusuf , Emre and Berfin
- Correlation analyses and Statistical Tests: Emre and Berfin
- Web interface  development and implementation: Seray
- Interactive form: Seray and Berfin
- Data Visualizations: Prashanth and Seray
- Read Me file: Prashanth

## Dataset source
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/code?datasetId=1936563&sortBy=voteCount







   

   
