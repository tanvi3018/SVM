#!/usr/bin/env python
# coding: utf-8

# ## Context:
# An MNC has thousands of employees spread out across the globe. The company believes in hiring the best talent available and retaining them for as long as possible. A huge amount of resources is spent on retaining existing employees through various initiatives. The Head of People Operations wants to bring down the cost of retaining employees. For this, he proposes limiting the incentives to only those employees who are at risk of attrition. As a recently hired Data Scientist, you have been asked to identify patterns in characteristics of employees who leave the organization. Also, you have to use this information to predict if an employee is at risk of attrition. This information will be used to target them with incentives.
# 
# ## Objective :
# 
# * To identify the different factors that drive attrition
# * To make a model to predict if an employee will attrite or not
# 
# 
# ## Dataset :
# 
# The data contains demographic details, work-related metrics and attrition flag.
# 
# * **EmployeeNumber** - Employee Identifier
# * **Attrition** - Did the employee attrite?
# * **Age** - Age of the employee
# * **BusinessTravel** - Travel commitments for the job
# * **DailyRate** - Data description not available**
# * **Department** - Employee Department
# * **DistanceFromHome** - Distance from work to home (in km)
# * **Education** - 1-Below College, 2-College, 3-Bachelor, 4-Master,5-Doctor
# * **EducationField** - Field of Education
# * **EnvironmentSatisfaction** - 1-Low, 2-Medium, 3-High, 4-Very High
# * **Gender** - Employee's gender
# * **HourlyRate** - Data description not available**
# * **JobInvolvement** - 1-Low, 2-Medium, 3-High, 4-Very High
# * **JobLevel** - Level of job (1 to 5)
# * **JobRole** - Job Roles
# * **JobSatisfaction** - 1-Low, 2-Medium, 3-High, 4-Very High
# * **MaritalStatus** - Marital Status
# * **MonthlyIncome** - Monthly Salary
# * **MonthlyRate** - Data description not available**
# * **NumCompaniesWorked** - Number of companies worked at
# * **Over18** - Over 18 years of age?
# * **OverTime** - Overtime?
# * **PercentSalaryHike** - The percentage increase in salary last year
# * **PerformanceRating** - 1-Low, 2-Good, 3-Excellent, 4-Outstanding
# * **RelationshipSatisfaction** - 1-Low, 2-Medium, 3-High, 4-Very High
# * **StandardHours** - Standard Hours
# * **StockOptionLevel** - Stock Option Level
# * **TotalWorkingYears** - Total years worked
# * **TrainingTimesLastYear** - Number of training attended last year
# * **WorkLifeBalance** - 1-Low, 2-Good, 3-Excellent, 4-Outstanding
# * **YearsAtCompany** - Years at Company
# * **YearsInCurrentRole** - Years in the current role
# * **YearsSinceLastPromotion** - Years since the last promotion
# * **YearsWithCurrManager** - Years with the current manager
# 
# **In the real world, you will not find definitions for some of your variables. It is a part of the analysis to figure out what they might mean.**

# ### Import the necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#to scale the data using z-score
from sklearn.preprocessing import StandardScaler

#to split the dataset
from sklearn.model_selection import train_test_split

#import logistic regression
from sklearn.linear_model import LogisticRegression

#to build SVM model
from sklearn.svm import SVC

#Metrics to evaluate the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#for tuning the model
from sklearn.model_selection import GridSearchCV

#to ignore warnings
import warnings
warnings.filterwarnings("ignore")


# ### Read the dataset

# In[ ]:


#read the dataset
df = pd.read_excel('HR_Employee_Attrition.xlsx')


# In[ ]:


df.sample(5)


# ### Printing the information

# In[ ]:


df.info()


# **Observations:**
# - There are 2940 observations and 34 columns.
# - All the column have 2940 non-null values i.e. there are no missing values in the data.

# **Let's check the unique values in each column**

# In[ ]:


#checking unique values in each column
df.nunique()


# **Observations:**
# - Employee number is an identifier which is unique for each employee and we can drop this column as it would not add any value to our analysis.
# - Over18 and StandardHours have only 1 unique value. These column will not add any value to our model hence we can drop them.
# - On the basis of number of unique values in each column and the data description, we can identify the continuous and categorical columns in the data.
# 
# Let's drop the columns mentioned above and define lists for numerical and categorical columns to apply explore them separately.

# In[ ]:


#dropping the columns
df=df.drop(['EmployeeNumber','Over18','StandardHours'],axis=1)


# In[ ]:


#Creating numerical columns
num_cols=['DailyRate','Age','DistanceFromHome','MonthlyIncome','MonthlyRate','PercentSalaryHike','TotalWorkingYears',
          'YearsAtCompany','NumCompaniesWorked','HourlyRate',
          'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TrainingTimesLastYear']

#Creating categorical variables
cat_cols= ['Attrition','OverTime','BusinessTravel', 'Department','Education', 'EducationField','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance',
           'StockOptionLevel','Gender', 'PerformanceRating', 'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus','RelationshipSatisfaction']


# ### Let's start with univariate analysis of numerical columns

# In[ ]:


#Checking summary statistics
df[num_cols].describe().T


# **Observations:**
# - **Average employee age is around 37 years**. It has a high range, from 18 years to 60, indicating good age diversity in the organization.
# - **At least 50% of the employees live within a 7 km radius** from the organization. However there are some extreme values, seeing as the maximum value is 29 km.
# - **The average monthly income of an employee is USD 6500.** It has a high range of values from 1K-20K USD, which is to be expected for any organization's income distribution. There is a big difference between the 3rd quartile value (around USD 8400) and the maximum value (nearly USD 20000), showing that the **company's highest earners have a disproportionately large income** in comparison to the rest of the employees. Again, this is fairly common in most organizations.
# - **Average salary hike of an employee is around 15%.** At least 50% of employees got a salary hike 14% or less, with the maximum salary hike being 25%.
# - Average number of years an employee is associated with the company is 7.
# - **On average, the number of years since an employee got a promotion is 2.18**. The majority of employees have been promoted since the last year.

# Let's explore these variables in some more depth by observing their distributions

# In[ ]:


#creating histograms
df[num_cols].hist(figsize=(14,14))
plt.show()


# **Observations:**
# 
# - **The age distribution is close to a normal distribution** with the majority of employees between the ages of 25 and 50.
# 
# - **The percentage salary hike is skewed to the right**, which means employees are mostly getting lower percentage salary increases.
# 
# - **MonthlyIncome and TotalWorkingYears are skewed to the right**, indicating that the majority of workers are in entry / mid-level positions in the organization.
# 
# - **DistanceFromHome also has a right skewed distribution**, meaning most employees live close to work but there are a few that live further away.
# 
# - **On average, an employee has worked at 2.5 companies.** Most employees have worked at only 1 company.
# 
# - **The YearsAtCompany variable distribution shows a good proportion of workers with 10+ years**, indicating a significant number of loyal employees at the organization.
# 
# - **The YearsInCurrentRole distribution has three peaks at 0, 2, and 7.** There are a few employees that have even stayed in the same role for 15 years and more.
# 
# - **The YearsSinceLastPromotion variable distribution indicates that some employees have not received a promotion in 10-15 years and are still working in the organization.** These employees are assumed to be high work-experience employees in upper-management roles, such as co-founders, C-suite employees etc.
# 
# - The distributions of DailyRate, HourlyRate and MonthlyRate appear to be uniform and do not provide much information. It could be that daily rate refers to the income earned per extra day worked while hourly rate could refer to the same concept applied for extra hours worked per day. Since these rates tend to be broadly similar for multiple employees in the same department, that explains the uniform distribution they show.

# ### Univariate analysis for categorical variables

# In[ ]:


#Printing the % sub categories of each category
for i in cat_cols:
    print(df[i].value_counts(normalize=True))
    print('*'*40)


# **Observations:**
# 
# - **The employee attrition rate is 16%.**
# - **Around 28% of the employees are working overtime.** This number appears to be on the higher side, and might indicate a stressed employee work-life.
# - 71% of the employees have traveled rarely, while around 19% have to travel frequently.
# - Around 73% of the employees come from an educational background in the Life Sciences and Medical fields.
# - Over 65% of employees work in the Research & Development department of the organization.
# - **Nearly 40% of the employees have low (1) or medium-low (2) job satisfaction** and environment satisfaction in the organization, indicating that the morale of the company appears to be somewhat low.
# - **Over 30% of the employees show low (1) to medium-low (2) job involvement.**
# - Over 80% of the employees either have none or very less stock options.
# - **In terms of performance ratings, none of the employees have rated lower than 3 (excellent).** About 85% of employees have a performance rating equal to 3 (excellent), while the remaining have a rating of 4 (outstanding). This could either mean that the majority of employees are top performers, or  the more likely scenario is that the organization could be highly lenient with its performance appraisal process.

# ### Bivariate and Multivariate analysis

# **We have analyzed different categorical and numerical variables.**
# 
# **Let's now check how does attrition rate is related with other categorical variables**

# In[ ]:


for i in cat_cols:
    if i!='Attrition':
        (pd.crosstab(df[i],df['Attrition'],normalize='index')*100).plot(kind='bar',figsize=(8,4),stacked=True)
        plt.ylabel('Percentage Attrition %')


# **Observations:**
#     
# - **Employees working overtime have more than a 30% chance of attrition**,
# which is very high compared to the 10% chance of attrition for employees who do not work extra hours.
# - As seen earlier, the majority of employees work for the R&D department. The chance of attrition there is ~15%
# - **Employees working as sales representatives have an attrition rate of around 40%** while HRs and Technicians have an attrition rate of around 25%. The sales and HR departments have higher attrition rates in comparison to an academic department like Research & Development, an observation that makes intuitive sense keeping in mind the differences in those job profiles. The high-pressure and incentive-based nature of Sales and Marketing roles may be contributing to their higher attrition rates.
# - **The lower the employee's job involvement, the higher their attrition chances appear to be, with 1-rated JobInvolvement employees attriting at 35%.** The reason for this could be that employees with lower job involvement might feel left out or less valued and have already started to explore new options, leading to a higher attrition rate.
# - **Employees at a lower job level also attrite more,** with 1-rated JobLevel employees showing a nearly 25% chance of attrition. These may be young employees who tend to explore more options in the initial stages of their careers.
# - **A low work-life balance rating clearly leads employees to attrite**; 30% of those in the 1-rated category show attrition.

# **Let's check the relationship between attrition and Numerical variables**

# In[ ]:


#Mean of numerical variables grouped by attrition
df.groupby(['Attrition'])[num_cols].mean()


# **Observations:**
# - **Employees leaving the company have a nearly 30% lower average income and 30% lesser work experience than those who are not.** These could be the employees looking to explore new options and/or increase their salary with a company switch.
# - **Employees showing attrition also tend to live 16% further from the office than those who are not**. The longer commute to and from work could mean they have to spend more time/money every day, and this could be leading to job dissatisfaction and wanting to leave the organization.

# **We have found out what kind of employees are leaving the company more.**
# 
# ### Let's check the relationship between different numerical variables

# In[ ]:


#plotting the correlation between numerical variables
plt.figure(figsize=(15,8))
sns.heatmap(df[num_cols].corr(),annot=True, fmt='0.2f', cmap='YlGnBu')


# **Observations:**
# 
# - **Total work experience, monthly income, years at company and years with current manager are highly correlated with each other and with employee age** which is easy to understand as these variables show an increase with age for most employees.
# - Years at company and years in current role are correlated with years since last promotion which means that the company is not giving promotions at the right time.

# **Now we have explored our data. Let's build the model**

# ## Model Building - Approach
# 1. Prepare data for modeling
# 2. Partition the data into train and test set.
# 3. Build model on the train data.
# 4. Tune the model if required.
# 5. Test the data on test set.

# ###  Preparing data for modeling

# **Creating dummy variables for categorical Variables**

# In[ ]:


#creating list of dummy columns
to_get_dummies_for = ['BusinessTravel', 'Department','Education', 'EducationField','EnvironmentSatisfaction', 'Gender',  'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus' ]

#creating dummy variables
df = pd.get_dummies(data = df, columns= to_get_dummies_for, drop_first= True)

#mapping overtime and attrition
dict_OverTime = {'Yes': 1, 'No':0}
dict_attrition = {'Yes': 1, 'No': 0}


df['OverTime'] = df.OverTime.map(dict_OverTime)
df['Attrition'] = df.Attrition.map(dict_attrition)


# **Separating the independent variables (X) and the dependent variable (Y)**

# In[ ]:


#Separating target variable and other variables
Y= df.Attrition
X= df.drop(columns = ['Attrition'])


# ### Scaling the data
# 
# The independent variables in this dataset have different scales. When features have different scales from each other, there is a chance that a higher weightage will be given to features that have a higher magnitude, and they will dominate over other features whose magnitude changes may be smaller but whose percentage changes may be just as significant or even larger. This will impact the performance of our machine learning algorithm, and we do not want our algorithm to be biased towards one feature.
# 
# The solution to this issue is **Feature Scaling**, i.e. scaling the dataset so as to give every transformed variable a comparable scale.
# 
# In this problem, we will use the **Standard Scaler** method, which centers and scales the dataset using the Z-Score.
# 
# It standardizes features by subtracting the mean and scaling it to have unit variance.
# 
# The standard score of a sample x is calculated as:
# 
# **z = (x - u) / s**
# 
# where **u** is the mean of the training samples (zero) and **s** is the standard deviation of the training samples.

# In[ ]:


#Scaling the data
sc=StandardScaler()
X_scaled=sc.fit_transform(X)
X_scaled=pd.DataFrame(X_scaled, columns=X.columns)


# ### Splitting the data into 80% train and 20% test set

# Some classification problems can exhibit a large imbalance in the distribution of the target classes: for instance there could be several times more negative samples than positive samples. In such cases it is recommended to use the **stratified sampling** technique to ensure that relative class frequencies are approximately preserved in each train and validation fold.

# In[ ]:


#splitting the data
x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)


# ### Model evaluation criterion
# 
# #### The model can make two types of wrong predictions:
# 1. Predicting an employee will attrite when the employee doesn't attrite
# 2. Predicting an employee will not attrite and the employee actually attrites
# 
# #### Which case is more important?
# * **Predicting that the employee will not attrite but the employee attrites** i.e. losing out on a valuable employee or asset. This would be considered a major miss for any employee attrition predictor, and is hence the more important case of wrong predictions.
# 
# #### How to reduce this loss i.e the need to reduce False Negatives?
# * **The company would want the Recall to be maximized**, the greater the Recall, the higher the chances of minimizing false negatives. Hence, the focus should be on increasing Recall (minimizing the false negatives) or in other words identifying the true positives (i.e. Class 1) very well, so that the company can provide incentives to control attrition rate especially for top-performers. This would help in optimizing the overall project cost towards retaining the best talent.

# Also, let's create a function to calculate and print the classification report and confusion matrix so that we don't have to rewrite the same code repeatedly for each model.

# In[ ]:


#creating metric function
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# #### Building the model
# 
# We will be building 2 different models:
# - **Logistic Regression**
# - **Support Vector Machine(SVM)**

# - The reported average includes the macro average which averages the unweighted mean per label, and the weighted average i.e. averaging the support-weighted mean per label.
# - In classification, the class of interest is considered the positive class. Here, the class of interest is 1 i.e. identifying the employees at risk of attrition.
# 
# **Reading the confusion matrix (clockwise):**
# 
# * True Negative (Actual=0, Predicted=0): Model predicts that an employee would not attrite and the employee does not attrite
# 
# * False Positive (Actual=0, Predicted=1): Model predicts that an employee would attrite but the employee does not attrite
# 
# * False Negative (Actual=1, Predicted=0): Model predicts that an employee would not attrite but the employee attrites
# 
# * True Positive (Actual=1, Predicted=1): Model predicts that an employee would attrite and the employee actually attrites

# ### Logistic Regression Model

# - Logistic Regression is a supervised learning algorithm which is used for **binary classification problems** i.e. where the dependent variable is categorical and has only two possible values. In logistic regression, we use the sigmoid function to calculate the probability of an event y, given some features x as:
# 
#                                           P(y)=1/(1 + exp(-x))

# In[ ]:


#fitting logistic regression model
lg=LogisticRegression()
lg.fit(x_train,y_train)


# **Checking model performance**

# - The reported average includes the macro average which averages the unweighted mean per label, and the weighted average i.e. averaging the support-weighted mean per label.
# - In classification, the class of interest is considered the positive class. Here, the class of interest is 1 i.e. identifying the employees at risk of attrition.
# 
# **Reading the confusion matrix (clockwise):**
# 
# * True Negative (Actual=0, Predicted=0): Model predicts that an employee would not attrite and the employee does not attrite
# 
# * False Positive (Actual=0, Predicted=1): Model predicts that an employee would attrite but the employee does not attrite
# 
# * False Negative (Actual=1, Predicted=0): Model predicts that an employee would not attrite but the employee attrites
# 
# * True Positive (Actual=1, Predicted=1): Model predicts that an employee would attrite and the employee actually attrites

# In[ ]:


#checking the performance on the training data
y_pred_train = lg.predict(x_train)
metrics_score(y_train, y_pred_train)


# In[ ]:


#checking the performance on the test dataset
y_pred_test = lg.predict(x_test)
metrics_score(y_test, y_pred_test)


# **Observations:**
# - **We are getting an accuracy of around 90%** on train and test dataset.
# - However, **the recall for this model is only around 50% for class 1 on train and 46% on test.**
# - As the recall is low, **this model will not perform well** in differentiating out those employees who have a high chance of leaving the company, meaning it will eventually not help in reducing the attrition rate.
# - As we can see from the Confusion Matrix, **this model fails to identify the majority of employees who are at risk of attrition.**

# In[ ]:





# ### Support Vector Machines (SVM)

# In[ ]:


#fitting SVM
svm = SVC(kernel = 'linear') #linear kernal or linear decision boundary
model = svm.fit(X = x_train, y = y_train)


# In[ ]:


y_pred_train_svm = model.predict(x_train)
metrics_score(y_train, y_pred_train_svm)


# In[ ]:


# Checking performance on the test data
y_pred_test_svm = model.predict(x_test)
metrics_score(y_test, y_pred_test_svm)


# #### Using RBF kernel

# In[ ]:


#fitting SVM
svm = SVC(kernel = 'rbf') #linear kernal or linear decision boundary
model = svm.fit(X = x_train, y = y_train)


# In[ ]:


y_pred_train_svm = model.predict(x_train)
metrics_score(y_train, y_pred_train_svm)


# In[ ]:


# Checking performance on the test data
y_pred_test_svm = model.predict(x_test)
metrics_score(y_test, y_pred_test_svm)


# #### Using polynomial

# In[ ]:


#fitting SVM
svm = SVC(kernel = 'poly', degree=3) #linear kernal or linear decision boundary
model = svm.fit(X = x_train, y = y_train)


# In[ ]:


y_pred_train_svm = model.predict(x_train)
metrics_score(y_train, y_pred_train_svm)


# * SVM model with rbf linear is not overfitting as the accuracy is around 80% for both train and test dataset
# * Recall for the model only around 50% which implies our model will not correctly predict the employees who are on the risk of attrite.
# * The precision is quite good and the model will help not five false positive and will save the cost and energy of the organization.
