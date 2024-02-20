# An Interactive Exploration of Heart Disease Indicators

## Description
According to the CDC, heart disease is a leading cause of death for people of most races in the U.S. About half of all Americans (47%) have at least one major risk factor for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicators include diabetes status, obesity (high BMI), level of physical activity, or alcohol consumption. Application of machine learning methods to detect patterns in the data can predict a patient's susceptibility to heart disease. This tool helps them to make informed decisions about lifestyle modifications to potentially reduce their risk of heart disease.

The [dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data) used for this project was taken from Kaggle.




## Functionalities
The project has the following functionalities, with a focus on leveraging external libraries to enhance our platform's capabilities:

### Data Sources and Retrieval

The data for our project is sourced from a comprehensive dataset available on Kaggle, originally collected by the CDC. We will use Python libraries such as requests for data retrieval, ensuring seamless integration into our platform.
This data set currently contains 246014 instances (after data cleaning). 36 relevant features have been extracted from a maximum of 40 in the total data.
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

The dataset was imported as a dataset using the Pandas as a CSV file in Python, which allows for efficient data manipulation and analysis. Our approach will ensure compatibility with various data analysis tools without the need for a separate database system.

### Data Pre-processing










