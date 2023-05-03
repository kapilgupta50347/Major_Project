# Python major project by kapil gupta (march batch-2023)
# CLASSIFICATION -> LOGISTIC REGRESSION

import pandas as pd

# 1. TAKE THE DATA AND CREATE A DATAFRAME
# As haberman.csv doesn't contain column names, so providing the names for columns in the dataset
col_Names=["patient_age", "year_of_operation", "positive_axillary_nodes", "survival_status"];

# To load the Haberman's Survival dataset into a pandas dataFrame
patient_df = pd.read_csv('https://raw.githubusercontent.com/bethusaisampath/Haberman-Cancer-Survival-Dataset/master/haberman.csv', names=col_Names);
# High level statistics of the dataset 

# To check the number of datapoints and number of attributes or features available in the dataset
print(patient_df.shape)
# To see column names in the dataset
print(patient_df.columns)
print(patient_df,'\n')

# 2. PREPROCESSING - FILTERING OF DATA, EDA
# performing Exploratory Data Analysis (EDA)

print(patient_df.info(),'\n')
print(patient_df.shape,'\n')
print(patient_df.size,'\n')

# Survival status = 1 and 2
# 1 = patient survived 5 years or longer
# 2 = patient died within 5 years

print(patient_df['survival_status'].value_counts(),'\n')

# 3. DATA VISUALIZATION - NOT REQUIRED HERE

# 4. DIVIDE THE DATA INTO INPUT AND OUTPUT
# input(x) is 2 dimensional and output(y) is 1 dimensional

x=patient_df.iloc[:,0:3].values
print(x,'\n')

y=patient_df.iloc[:,3].values
print(y,'\n')

# 5. TRAIN AND TEST VARIABLES

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.6,random_state=2) #train size is 60%

# 6. NORMALIZE THE DATA(TO BE DONE ONLY ON INPUTS OF MULTIVARIATE DATASETS)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

# 7. RUN A REGRESSOR/CLASSIFIER/CLUSTERER

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

# 8. FIT THE MODEL

model.fit(x_train,y_train)

# 9. PREDICT THE OUTPUT

y_pred=model.predict(x_test)
print(y_pred,'\n')
print(y_test,'\n')

# 10. ACCURACY SCORE

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test)*100)
