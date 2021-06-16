# Predicting Length of Stay (LOS) in a Hospital

## Author: Nischay Thapa

Hospitals are constantly challenged to provide timely patient care while maintaining high resource utilization.   While this challenge has been around for many years,  the recent COVID-19 pandemic has increased its prominence.  For a hospital, the ability to predict the length of stay (LOS) of a  patient as early as possible (at the admission stage) is very useful in managing its resources. Particularly, we are interested in developing a machine learning model to predict if a patient will be discharged from a hospital early or will stay in a hospital for an extended period based on several attributes related to patient characteristics, diagnoses, treatments, services, hospital charges and patients socio-economic background. We formulate this task as a binary classification and predict whether a patient will be discharged within 3 days or stay longer.

## Dataset Description

The original data is from [HealthData: Hospital Inpatient Discharges (SPARCS De-Identified)](https://healthdata.gov). The data provided is based on this, with some modifications.

The Statewide Planning and Research Cooperative System (SPARCS) Inpatient De-identified File contains discharge level detail on patient characteristics, diagnoses, treatments, services, and charges.

### Licence agreement: 

Sharing or distributing this data or using this data for any other commercial or non-commercial purposes is prohibited.


### Data Fields

| Column   Name                | Attribute/Target | Description                                                                                                                                                                                                  |
|------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ID                           | N/A              | Unique number to represent patient ID                                                                                                                                                                        |
| HealthServiceArea            | N/A              | A description of the Health Service Area (HSA) in which the hospital is located. Capital/Adirondack, Central NY, Finger   Lakes, Hudson Valley, Long Island, New York City, Southern Tier, Western NY.       |
| Gender                       | Attribute 1      | Patient gender:   (M) Male, (F) Female, (U) Unknown.                                                                                                                                                         |
| Race                         | Attribute 2      | Patient race. Black/African American, Multi, Other Race, Unknown, White. Other Race   includes Native Americans and Asian/Pacific Islander.                                                                  |
| TypeOfAdmission              | Attribute 3      | A description of   the manner in which the patient was admitted to the health care facility:   Elective, Emergency, Newborn, Not Available, Trauma, Urgent.                                                  |
| CCSProcedureCode             | Attribute 4      | AHRQ Clinical Classification Software (CCS) ICD-9 Procedure Category Code                                                                                                                                    |
| APRSeverityOfIllnessCode     | Attribute 5      | All Patient  Refined Severity of Illness (APR SOI) Description: Minor (1), Moderate (2),   Major (3), Extreme (4)                                                                                            |
| PaymentTypology              | Attribute 6      | A description of the type of payment for this occurrence.                                                                                                                                                    |
| BirthWeight                  | Attribute 7      | The neonate birth weight in grams; rounded to nearest 100g.                                                                                                                                                  |
| EmergencyDepartmentIndicator | Attribute 8      | Emergency Department Indicator is set based on the submitted revenue codes. If the   record contained an Emergency Department revenue code of 045X, the indicator   is set to "Y", otherwise it will be “N”. |
| AverageCostInCounty          | Attribute 9      | Average hospitalization Cost In County of the patient                                                                                                                                                        |
| AverageChargesInCounty       | Attribute 10     | Average medical Charges In County of the patient                                                                                                                                                             |
| AverageCostInFacility        | Attribute 11     | Average Cost In Facility                                                                                                                                                                                     |
| AverageChargesInFacility     | Attribute 12     | Average Charges In Facility                                                                                                                                                                                  |
| AverageIncomeInZipCode       | Attribute 13     | Average Income In Zip Code                                                                                                                                                                                   |
| LengthOfStay                 | target           | The total number  of patient days at an acute level and/or other than acute care level. Need to be transformed to match the task class 0 id LengthOfStay < 4 and class 1 otherwise                           |


## Exploratory Data Analysis

## Target 

![](https://github.com/nischaybikramthapa/predicting-hospital-los/blob/main/images/target.JPG)

The target attribute is highly imablanced with majority of patient discharged within 3 days period. A failure in choosing a appropriate evaluation metric can lead us to poor model performance. Hence, we address this issue and propose a suitable evaluation strategy in the later section of this analysis.

### Data Distribution I
![](https://github.com/nischaybikramthapa/predicting-hospital-los/blob/main/images/eda.JPG)

Looking at the above histograms, <code> Gender, Race, TypeOfAdmissions, CCSProcedureCode, PaymentTypology</code> and <code>EmergencyDepartmentNumber</code> represent categorical variables where a patient fall under each group based on thier characteristics. <code>APRSeverityOfIllnessCode</code> on the other hand is an ordinal attribute where severity of illness ranges from 1 being minor to 4 as extreme. There are many numerical variables such as <code>Birthweight, AverageCostInCount, AverageChargesInCounty,AverageCostInFacility, AverageChargesInFacility </code> and <code>AverageIncomeInZipCode </code>that are associated with weight and cost incurred while the patient is admitteed to a hospital. The distrbution of these numeric variables are positive and skewed to the right. If we use parametric models like logistic regression, we would want to normalise the datasets to get better results without loosing key information. Also, health service areas in traning set and test set are different. We will use this group to split our data so that we have indepedent results for different areas of hospitals. 


### Data Distribution II

We hypothesized that some numerical attributes might have different range of values showing discriminative ability on our target. To validate this, we create multiple box plots for each numeric variables with the target. The difference in median values in most attributes indicates variability that can somewhat help to distinguish the length of stay in a patient, however, the graphical evidence are not significant to hold strongly onto this assumption. In addition, the range of each numerical attributes vary significantly. Features such as BirthWeight, AverageCostInCounty, and AverageCostInFacility consists of outliers that can affect model performance. After a closer look on the largest outlier point from BirthWeight, the data point seems to be an obvious value which can be handled with appropriate transformation or scaling techniques. Although it entirely depends on the types of model we choose, data pre-processing allow us to avoid noise and inconsistencies further improving the performance of a predictive model.

![](https://github.com/nischaybikramthapa/predicting-hospital-los/blob/main/images/box.JPG)

### Correlation between attributes
![](https://github.com/nischaybikramthapa/predicting-hospital-los/blob/main/images/corr.JPG)

Features such as  AverageChargesInCounty, AverageChargesInFacility and AverageCostInCounty  are highy correlated. While these features represent cost of hospitalisation based on facilities and county, a new feature combining these attributes can be used in the model. However, we limit our analysis using all the attributes ignoring these associations. Further, we explore each pair of columns separated with class labels to understand thier relationships.

### EDA Summary
![](https://github.com/nischaybikramthapa/predicting-hospital-los/blob/main/images/eda_all.JPG)

From the Exploratory Data Analysis, the following observations and assumptions are made:

* Since there are few categories within 5 nominal variables, one-hot encoding representation of these features can be used in the model building process.
* The pair plot shows a non-linear relationship between target and classes so we use different linear and non-linear classifiers for predictions.
* As our class labels are highly imbalanced, we evaluate model performance based on macro-average f1-measure treating both classes equally. 
* We validate the performance based on Hold-Out set to avoid bias in optimisation of majority class and find optimal hyperparameters.

## Experiment 

### Result

| Model   | Train F1  | Train BCE  | Val F1 | Val BCE | 
| :-----: | :-------: | :--------: | :----: | :-----: |
| balanced linear | 0.600       | 11.933        |    0.644  | 13.758      |
| balanced_linear_lasso | 0.601       | 11.983        |    0.644  | 13.783    |
| <b> balanced_poly2_lasso |  <b> 0.611     | <b> 10.976      | <b>   0.640  | <b> 14.332     </b> |
| adaboost| 0.608      | 12.576      |    0.623 | 14.636    |
| decision tree | 0.817       | 2.465        |    0.569  | 16.985      |
| random forest | 0.741     | 7.131      |    0.588 | 17.215   |

  ### Feature Importances
  
  ![fimp](https://user-images.githubusercontent.com/46451619/122047004-f637e400-ce22-11eb-88b5-4304133899fb.JPG)

## Conclusion

 Overall, Polynomial Logistic Regression with a lasso penalty performed comparatively better than other classifers. We limit our analysis on machine learning algorithsm to have better explainability on how the models work and what features are important indicator for determining length of stay. Based on the results, our model is fair for all gender, race and hospital areas. This analysis could be extended as per requirement such as predicting more number of longer hospitals. However, this depends entirely on the business problem.
  
## Code
Detail analysis of this project is available [here](https://github.com/nischaybikramthapa/predicting-hospital-los/blob/main/Notebook.ipynb")
 
