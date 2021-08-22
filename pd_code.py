import numpy as np
import pandas as pd
import os
os.chdir('C:/Users/Acer/Desktop/Major 2021')

loan_data_backup = pd.read_csv('Loan_default.csv')

loan_data = loan_data_backup.copy()
loan_data
#loan_data.replace('nan','')
#loan_data['Gender'] = loan_data['Gender'].fillna('')

loan_data.head()
loan_data.tail()

#displays all columns names.
loan_data.columns.values

#displays column names, complete (non-missing) cases per column, and datatype per column.
loan_data.info()
loan_data.describe()


#CLEANING THE DATA - Dealing with missing values

#It returns 'false' if a value is not missing and 'true' if a value is missing, for each value in a dataframe.
loan_data.isnull()

#Sets the pandas dataframe options to display all columns/rows.
loan_data.isnull().sum()

# 1st variable - Gender
loan_data['Gender'].unique()
#loan_data['Gender'] = loan_data['Gender'].str.replace('nan','')
#loan_data['Gender'] = loan_data['Gender'].replace('nan','')
Gender_map = {'Male': 1, 'Female': 0}
loan_data['Gender1'] = loan_data['Gender'].map(Gender_map)
a = loan_data['Gender1'].mode()
a
loan_data['Gender1'] = loan_data['Gender1'].replace(np.nan, 1)
loan_data.info()
loan_data.isnull().sum()
#2nd variable - married
loan_data['Married'].unique()
Married_map = {'Yes': 1, 'No': 0}
loan_data['Married1'] = loan_data['Married'].map(Married_map)
a = loan_data['Married1'].mode()
a
loan_data['Married1'] = loan_data['Married1'].replace(np.nan, 1)
loan_data.info()
loan_data.isnull().sum()

#3rd variable - dependents
loan_data['Dependents'].unique()
Dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
loan_data['Dependents1'] = loan_data['Dependents'].map(Dependents_map)
a = loan_data['Dependents1'].mode()
a
loan_data['Dependents1'] = loan_data['Dependents1'].replace(np.nan, 0)
loan_data.info()
loan_data.isnull().sum()


# 4th variable - Education
loan_data['Education'].unique()
Education_map = {'Graduate': 1, 'Non Graduate': 0}
loan_data['Education1'] = loan_data['Education'].map(Education_map)
a = loan_data['Education1'].mode()
a
loan_data['Education1'] = loan_data['Education1'].replace(np.nan, 1)
loan_data.info()
loan_data.isnull().sum()

# 5th Variable - Self Employed
loan_data['Self_Employed'].unique()
Self_Employed_map = {'Yes':1, 'No':0}
loan_data['Self_Employed1'] = loan_data['Self_Employed'].map(Self_Employed_map)
a = loan_data['Self_Employed1'].mode()
a
loan_data['Self_Employed1'] = loan_data['Self_Employed1'].replace(np.nan, 0)
loan_data.info()
loan_data.isnull().sum()

# 6th Variable - LoanAmount
loan_data.info()
loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean(), inplace=True)
loan_data.isnull().sum()

#7th Variable - Loan_Amount_Term
loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mean(), inplace=True)
loan_data.isnull().sum()

#8th Variable - Credit_History
loan_data['Credit_History'].fillna(0, inplace=True)
loan_data.isnull().sum()

#We will now delete the old variables
loan_data.drop('Gender', axis=1, inplace = True)
loan_data.drop('Education', axis=1, inplace = True)
loan_data.drop('Married', axis=1, inplace = True)
loan_data.drop('Dependents', axis=1, inplace = True)
loan_data.drop('Self_Employed', axis=1, inplace = True)

#Rename Gender1 to Gender
# changing cols with rename()
loan_data1 = loan_data.rename(columns = {"Gender1": "Gender",
                                         "Education1": "Education",
                                         "Married1": "Married",
                                         "Dependents1": "Dependents",
                                         "Self_Employed1": "Self_Employed"})
loan_data1.info()
loan_data1.isnull().sum()

# DISCRETE VARIABLES

# Displays column names, complete (non-missing) cases per column, and datatype per column
loan_data1.info()
loan_data1['Loan_Amount_Term'].unique()
loan_data1['Loan_Amount_Term'].mean()


#1st discrete variable: GRADE
# Create dummy variables from a variable


# We create dummy variables from all 8 original independent variables, and save them into a list.
# Note that we are using a particular naming convention for all variables: original variable name, colon, category name.
loan_data_dummies1 = [pd.get_dummies(loan_data1['Gender'], prefix = 'Gender', prefix_sep = ':'),
                      pd.get_dummies(loan_data1['Married'], prefix = 'Married', prefix_sep = ':'),
                      pd.get_dummies(loan_data1['Dependents'], prefix = 'Dependents', prefix_sep = ':'),
                      pd.get_dummies(loan_data1['Education'], prefix = 'Education', prefix_sep = ':'),
                      pd.get_dummies(loan_data1['Self_Employed'], prefix = 'Self_Employed', prefix_sep = ':'),
                      pd.get_dummies(loan_data1['Credit_History'], prefix = 'Credit_History', prefix_sep = ':'),
                      pd.get_dummies(loan_data1['Location'], prefix = 'Location', prefix_sep = ':'),
                      pd.get_dummies(loan_data1['Loan_Amount_Term'], prefix = 'Loan_Amount_Term', prefix_sep = ':')]

#We concatenate the dummy variables and this turns them into a dataframe.
loan_data_dummies1 = pd.concat(loan_data_dummies1, axis = 1)

#Returns the type of the variable.
type(loan_data_dummies1)

# Concatenates two dataframes.
# Here we concatenate the dataframe with the original data with the dataframe with dummy variables, along the columns.
loan_data1 = pd.concat([loan_data1, loan_data_dummies1], axis = 1)

# Displays all column names.
loan_data1.columns.values

# Good/Bad definition
# We create a new variable that has the value of '0' if a condition is met, and the value of '1' if it is not met.
Loan_Status_map = {'Y': 1, 'N': 0}
loan_data1['target'] = loan_data1['Loan_Status'].map(Loan_Status_map)

# Displays unique values of a column.
loan_data1['target'].unique()
# Calculates the number of observations for each unique value of a variable.
loan_data1['target'].value_counts()
loan_data1.drop('Loan_Status', axis=1, inplace = True)

loan_data1['Location'].unique()
Location_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
loan_data1['Location1'] = loan_data1['Location'].map(Location_map)
loan_data.drop('Location', axis=1, inplace = True)

loan_data1 = loan_data1.rename(columns = {"Location1": "Location"})
loan_data1.info()

# Dropping the original variables for which we have dummies now
loan_data1.drop('Gender', axis=1, inplace = True)
loan_data1.drop('Education', axis=1, inplace = True)
loan_data1.drop('Married', axis=1, inplace = True)
loan_data1.drop('Dependents', axis=1, inplace = True)
loan_data1.drop('Self_Employed', axis=1, inplace = True)
loan_data1.drop('Loan_Amount_Term', axis=1, inplace = True)
loan_data1.drop('Location', axis=1, inplace = True)
loan_data1.drop('Credit_History', axis=1, inplace = True)
#loan_data1.drop('Self_Employed', axis=1, inplace = True)
loan_data1.info()

#Dropping the reference variables now for discrete variables
loan_data1.drop('Gender:0.0', axis=1, inplace = True)
#loan_data1.drop('Education:0.0', axis=1, inplace = True)
loan_data1.drop('Married:0.0', axis=1, inplace = True)
loan_data1.drop('Dependents:0.0', axis=1, inplace = True)
loan_data1.drop('Self_Employed:0.0', axis=1, inplace = True)
loan_data1.drop('Credit_History:0.0', axis=1, inplace = True)
loan_data1.drop('Location:Rural', axis=1, inplace = True)
#loan_data1.drop('Gender', axis=1, inplace = True)

#Dropping Loan_ID variable as well
loan_data1.drop('Loan_ID', axis=1, inplace = True)

loan_data1.info()
loan_data1.head()

#Splitting the data
from sklearn.model_selection import train_test_split
#We split two dataframes with inputs and targets, each into a train and test dataframe, and store them in variables.
#The time we split the size of the test dataset to be 30%.
# Respectively, the size of the train dataset becomes 70%.
# We also set a specific random state
# This would allow us to perform the exact same split multiple times.
# This means, to assign the exact same observations to the train and test datasets.
loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data1.drop('target', axis = 1), loan_data1['target'], test_size = 0.3, random_state = 42)

#Displays the size of the dataframe.
loan_data_inputs_train.shape
loan_data_targets_train.shape
loan_data_inputs_test.shape
loan_data_targets_test.shape


loan_data1.info()
#Dataa preparation
df_inputs_test = loan_data_inputs_test
#df_inputs_prepr + loan_data_inputs_test
df_inputs_prepr = loan_data_inputs_train

#df_targets_prepr = lean_data_targets_test
df_targets_prepr = loan_data_targets_train

#Computing Information Value for GRADE Variable
# Displays unique values of a column.
df_inputs_prepr['Gender:1.0'].unique()

#Concatenates two dataframes along the columns.
df1 = pd.concat([df_inputs_prepr['Gender:1.0'], df_targets_prepr], axis = 1)
df1.head()

#Groups the data according to a criterion contained in one column.
# Does not turn the names of the values of the criterion as indexes.
# Aggregates the data in another column, using a selected function.
# In this specific case, we group by the column with index 0 and we aggregate the values of the column with index 1.
# More specifically, we count them.
# In other words, we count the values in the column with index 1 for each value of the column with index 0.
df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count()

# Groups the data according to a criterion contained in one column.
# Does not turn the names of the values of the criterion as indexes.
# Aggregates the data in another column, using a selected function.
# Here we calculate the mean of the values in the column with index 1 for each value of the column with index 0.
df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()

#Concatenates two dataframes along the columns.
df1 = pd.concat([df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count(),
                 df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()], axis = 1)
df1

# Selects only columns with specific indexes.
df1 = df1.iloc[:, [0, 1, 3]]
df1

#Changes the names of the columns of a dataframe.
df1.columns = [df1.columns.values[0], 'n_obs', 'prop_good']
df1

# We divide the values of one column by the values of another column and save the result in a new variable.
df1['prop_n_obs'] = df1['n_obs']/df1['n_obs'].sum()
df1

# We multiply the values of one column by the values of another column and save the result in a new variable.
df1['n_good'] = df1['prop_good']*df1['n_obs']
df1['n_bad'] = (1 - df1['prop_good'])*df1['n_obs']
df1

df1['prop_n_good'] = df1['n_good'] / df1['n_good'].sum()
df1['prop_n_bad'] = df1['n_bad'] / df1['n_bad'].sum()
df1

# We take the natural logarithms of a variable and save the result in a new variable.
df1['WoE'] = np.log(df1['prop_n_good'] / df1['prop_n_bad'])
df1


# Sorts a dataframe by the values of a given column.
df1 = df1.sort_values(['WoE'])
# We reset the index of a dataframe and overwrite it.
df1 = df1.reset_index(drop = True)
df1

# We take the difference between two subsequent values of a column. Then, we take the absolute value of the result.
df1['diff_prop_good'] = df1['prop_good'].diff().abs()
# We take the difference between two subsequent values of a column. Then, we take the absolute value of the result.
df1['diff_WoE'] = df1['WoE'].diff().abs()
df1

df1['IV'] = (df1['prop_n_good'] - df1['prop_n_bad'])*df1['WoE']
df1['IV'] = df1['IV'].sum()
# We sum all values of a given column.
df1
# This gives the IV for Gender:1.0 Variable: 0.602922


# Creating a function to automate the above process of computing information value

# WoE function for discrete unordered variables
# Here we combine all of the operations above in a function.
def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs']/df['n_obs'].sum()
    df['n_good'] = df['prop_good']*df['n_obs']
    df['n_bad'] = (1 - df['prop_good'])*df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df
# The function takes 3 arguments: a dataframe, a string, and a dataframe.
#The function returns a dataframe as a result.
df_inputs_prepr.columns.values
df_temp = woe_discrete(df_inputs_prepr, 'Gender:1.0', df_targets_prepr)
df_temp
#0.000602
df_temp = woe_discrete(df_inputs_prepr, 'Married:1.0', df_targets_prepr)
df_temp
#0.045793

df_temp = woe_discrete(df_inputs_prepr, 'Dependents:1.0', df_targets_prepr)
df_temp  
#0.018841

df_temp = woe_discrete(df_inputs_prepr, 'Dependents:2.0', df_targets_prepr)
df_temp
#0.018651

df_temp = woe_discrete(df_inputs_prepr, 'Dependents:3.0', df_targets_prepr)
df_temp
#0.010003

df_temp = woe_discrete(df_inputs_prepr, 'Education:1.0', df_targets_prepr)
df_temp
#0.01673

df_temp = woe_discrete(df_inputs_prepr, 'Self_Employed:1.0', df_targets_prepr)
df_temp
#0.0000298

df_temp = woe_discrete(df_inputs_prepr, 'Credit_History:1.0', df_targets_prepr)
df_temp
#0.741148

df_temp = woe_discrete(df_inputs_prepr, 'Location:Semiurban', df_targets_prepr)
df_temp
#0.180498

df_temp = woe_discrete(df_inputs_prepr, 'Location:Urban', df_targets_prepr)
df_temp
#0.048117
df_inputs_prepr.columns.values

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:12.0', df_targets_prepr)
df_temp
#We can delete the above one

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:36.0', df_targets_prepr)
df_temp

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:60.0', df_targets_prepr)
df_temp

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:84.0', df_targets_prepr)
df_temp
#0.000218

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:120.0', df_targets_prepr)
df_temp

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:180.0', df_targets_prepr)
df_temp
#0.010994

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:240.0', df_targets_prepr)
df_temp
#0.000218
df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:300.0', df_targets_prepr)
df_temp
#0.005497
df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:342.0', df_targets_prepr)
df_temp
#0.002621

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:360.0', df_targets_prepr)
df_temp
#0.05809

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:480.0', df_targets_prepr)
df_temp
#0.079536

df_temp = woe_discrete(df_inputs_prepr, 'Loan_Amount_Term:342.0', df_targets_prepr)
df_temp
#0.002621


# We execute the function we defined with the necessary argument: a dataframe, a string, and a dataframe.
# We store the result in a dataframe.
df_temp

# Visualizing results

import matplotlib.pyplot as plt
import seaborn as sns
# Imports the libraries we need.
sns.set()
# We set the default style of the graphs to the seaborn style.

# Below we define a function that takes 2 arguments: a dataframe and a number.
# The number parameter has a default value 0.
# This means that if we call the function and omit the number parameter, it will be executed with it having a value of 0.
# The function displays a graph.
def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    # Turns the values of the column with the index 0 to strings, makes an array from these strings, and passes it to variable x.
    y = df_WoE['WoE']
    # Selects a column with label 'WoE' and passes it to variable y.
    plt.figure(figsize=(18, 6))
    # Sets the graph size to width 18 x height 6.
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    # Plots the datapoints with coordinates variable x on the x-axis and variable y on the y-axis.
    # Sets the marker for each datapoint to a circle, the style line between the points to dashed, and color to black.
    plt.xlabel(df_WoE.columns[0])
    # Names the x-axis with the name of the column with index 0.
    plt.ylabel('Weight of Evidence')
    # Names the y-axis 'Weight of Evidence'.
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    # Names the graph 'Weight of Evidence by ' the name of the column with index 0.
    plt.xticks(rotation = rotation_of_x_axis_labels)
    # Rotates the labels of the x-axis a predefined number of degrees.
    
plot_by_woe(df_temp)


# Computing IV for continuous variables
# CONTINUOUS VARIABLES
# WoE function for ordered discrete and continuous variables
def woe_ordered_continuous(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs']/df['n_obs'].sum()
    df['n_good'] = df['prop_good']*df['n_obs']
    df['n_bad'] = (1 - df['prop_good'])*df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

df_inputs_prepr.columns.values

#df_inputs_prepr['emp_length_int'].unique()
df_inputs_prepr['ApplicantIncome'].head()
df_inputs_prepr['CoapplicantIncome'].head()
df_inputs_prepr['LoanAmount'].head()


df_temp = woe_ordered_continuous(df_inputs_prepr, 'ApplicantIncome', df_targets_prepr)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'CoapplicantIncome', df_targets_prepr)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'LoanAmount', df_targets_prepr)

# We calculate weight of evidence.
df_temp
plot_by_woe(df_temp)

# We plot the weight of evidence values.
#1st variable - Applicant Income
df_inputs_prepr['ApplicantIncome']
df_inputs_prepr['ApplicantIncome_factor'] = pd.cut(df_inputs_prepr['ApplicantIncome'], 50)
#50 bins were not making any sense
df_inputs_prepr['ApplicantIncome_factor'] = pd.cut(df_inputs_prepr['ApplicantIncome'], 10)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['ApplicantIncome_factor'] = pd.cut(df_inputs_prepr['ApplicantIncome'], 5)

df_temp = woe_ordered_continuous(df_inputs_prepr, 'ApplicantIncome_factor', df_targets_prepr)
# We calculate weight of evidence.
df_temp

plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.
df_inputs_prepr['ApplicantIncome:<16320'] = np.where((df_inputs_prepr['ApplicantIncome'] <= 16320), 1, 0)
df_inputs_prepr['ApplicantIncome:16320-32490'] = np.where((df_inputs_prepr['ApplicantIncome'] > 16320) & (df_inputs_prepr['ApplicantIncome'] <= 32490), 1, 0)
df_inputs_prepr['ApplicantIncome:>32490'] = np.where((df_inputs_prepr['ApplicantIncome'] > 32490), 1, 0)

df_temp = woe_ordered_continuous(df_inputs_prepr, 'ApplicantIncome:<16320', df_targets_prepr)
df_temp
#0.000741

df_temp = woe_ordered_continuous(df_inputs_prepr, 'ApplicantIncome:16320-32490', df_targets_prepr)
df_temp
#0.009166

df_temp = woe_ordered_continuous(df_inputs_prepr, 'ApplicantIncome:>32490', df_targets_prepr)
df_temp
#0.007989

#2nd variable - Coapplicant Income
df_inputs_prepr['CoapplicantIncome']
df_inputs_prepr['CoapplicantIncome_factor'] = pd.cut(df_inputs_prepr['CoapplicantIncome'], 50)

#50 bins were not making any sense
df_inputs_prepr['CoapplicantIncome_factor'] = pd.cut(df_inputs_prepr['CoapplicantIncome'], 10)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['CoapplicantIncome_factor'] = pd.cut(df_inputs_prepr['ApplicantIncome'], 5)

df_temp = woe_ordered_continuous(df_inputs_prepr, 'CoapplicantIncome_factor', df_targets_prepr)
# We calculate weight of evidence.
df_temp

plot_by_woe(df_temp, 90)

df_inputs_prepr['CoapplicantIncome:<6767.4'] = np.where((df_inputs_prepr['CoapplicantIncome'] <= 6767.4), 1, 0)
df_inputs_prepr['CoapplicantIncome:6767.4-13534.8'] = np.where((df_inputs_prepr['CoapplicantIncome'] > 6767.4) & (df_inputs_prepr['CoapplicantIncome'] <= 13534.8), 1, 0)
df_inputs_prepr['CoapplicantIncome:>13534.8'] = np.where((df_inputs_prepr['CoapplicantIncome'] > 13534.8), 1, 0)

df_temp = woe_ordered_continuous(df_inputs_prepr, 'CoapplicantIncome:<6767.4', df_targets_prepr)
df_temp
#0.000337

df_temp = woe_ordered_continuous(df_inputs_prepr, 'CoapplicantIncome:6767.4-13534.8', df_targets_prepr)
df_temp
#0.016762

df_temp = woe_ordered_continuous(df_inputs_prepr, 'CoapplicantIncome:>13534.8', df_targets_prepr)
df_temp
#0.01955

#3rd variable - LoanAmount
df_inputs_prepr['LoanAmount']
df_inputs_prepr['LoanAmount_factor'] = pd.cut(df_inputs_prepr['LoanAmount'], 50)

#50 bins were not making any sense
df_inputs_prepr['LoanAmount_factor'] = pd.cut(df_inputs_prepr['LoanAmount'], 10)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['LoanAmount_factor'] = pd.cut(df_inputs_prepr['LoanAmount'], 5)

df_temp = woe_ordered_continuous(df_inputs_prepr, 'LoanAmount_factor', df_targets_prepr)
# We calculate weight of evidence.
df_temp

plot_by_woe(df_temp, 90)

df_inputs_prepr['LoanAmount:<290.2'] = np.where((df_inputs_prepr['LoanAmount'] <= 290.2), 1, 0)

df_inputs_prepr['LoanAmount:290.2-426.8'] = np.where((df_inputs_prepr['LoanAmount'] > 290.2) & (df_inputs_prepr['LoanAmount'] <= 426.8), 1, 0)
df_inputs_prepr['LoanAmount:>426.8'] = np.where((df_inputs_prepr['LoanAmount'] > 426.8), 1, 0)

df_temp = woe_ordered_continuous(df_inputs_prepr, 'LoanAmount:<290.2', df_targets_prepr)
df_temp
#0.000089

df_temp = woe_ordered_continuous(df_inputs_prepr, 'LoanAmount:290.2-426.8', df_targets_prepr)
df_temp
#0.002759

df_temp = woe_ordered_continuous(df_inputs_prepr, 'LoanAmount:>426.8', df_targets_prepr)
df_temp
#0.002928

#Final variables on the basis of IV are
#Married:1.0
#Credit_History:1.0
#Location:Semiurban
#Location:Urban
#Loan_Amount_Term:480.0



cols1=[ 'Married:1.0', 'Credit_History:1.0', 'Location:Semiurban', 'Location:Urban', 'Loan_Amount_Term:480.0']

X1=df_inputs_prepr[cols1]

a =X1.describe()
a
cols1=[ 'Married:1.0', 'Credit_History:1.0', 'Location:Semiurban', 'Location:Urban', 'Loan_Amount_Term:480.0']
cols2=[ 'Married:1.0', 'Credit_History:1.0', 'Location:Urban', 'Loan_Amount_Term:480.0']

X2=df_inputs_prepr[cols2]

#Correlation matrix
corr_matrix = X2.corr()

a =X2.describe()
a

#Computing VIF
#Imports
import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


#For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif["features"] = X2.columns
vif.round(1)

y=df_targets_prepr
y

#Implementing the model
#module is used
import statsmodels.api as sm
#y.dtypes
logit_model=sm.Logit(y,X2)
result=logit_model.fit()
print(result.summary2())

inputs_train = X2


#PD Model Estimation
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
reg = LogisticRegression()
pd.options.display.max_rows = None

reg.fit(inputs_train, loan_data_targets_train)




reg.intercept_

reg.coef_


feature_name = inputs_train.columns.values


summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)

summary_table['Coefficients'] = np.transpose(reg.coef_)


summary_table.index = summary_table.index + 1

summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

summary_table = summary_table.sort_index()

summary_table




from sklearn import linear_model
import scipy.stats as stat

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)
        
    def fit(self,X,y):
        self.model.fit(X,y)
        
        
        denom = (2.0* (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values
        
reg = LogisticRegression_with_p_values()

reg.fit(inputs_train, loan_data_targets_train)




summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

p_values = reg.p_values


p_values = np.append(np.nan, np.array(p_values))

summary_table['p_values'] = p_values
summary_table
summary_table.to_csv('summary_table.csv')



cols3=['Married:1.0','Credit_History:1.0']

X3=df_inputs_prepr[cols3]
inputs_train = X3

reg2 = LogisticRegression_with_p_values()
reg2.fit(inputs_train, loan_data_targets_train)
feature_name = inputs_train.columns.values


summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg2.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg2.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

p_values = reg2.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table


summary_table.to_csv('summary_table.csv')

#Final Model
logit_model=sm.Logit(y,X3)
result=logit_model.fit()
print(result.summary2())


df_inputs_test


cols3=['Married:1.0','Credit_History:1.0']
X_test=df_inputs_test[cols3]
X_test

loan_data_targets_test.shape
Y_test_data = loan_data_targets_test


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X3,y)


predictions = logmodel.predict(X_test)

y_pred_logistic = logmodel.decision_function(X_test)





from sklearn.metrics import classification_report
classification_report(Y_test_data, predictions)


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test_data, predictions)



from sklearn.metrics import accuracy_score
accuracy_score(Y_test_data, predictions)
#77.29%
#------------------Logistic Regression ends here--------------------


from sklearn.svm import SVC
model_SVC = SVC(kernel = 'rbf', random_state = 42)
model_SVC.fit(X3,y)

y_pred_svm = model_SVC.decision_function(X_test)




from sklearn.metrics import roc_curve, auc

logistic_fpr, logistic_tpr, threshold = roc_curve(Y_test_data, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

svm_fpr, svm_tpr, threshold = roc_curve(Y_test_data, y_pred_svm)
auc_svm = auc(svm_fpr, svm_tpr)

plt.figure(figsize = (5,5), dpi = 100)
plt.plot(svm_fpr, svm_tpr, linestyle = '-', label = 'SVM (auc = %0.3f)' % auc_svm)
plt.plot(logistic_fpr, logistic_tpr, marker = '.', label = 'Logistic (auc = %0.3f)' % auc_logistic)
plt.xlabel ('False Positive Rate')
plt.ylabel ('True Positive Rate')
plt.legend()
plt.show()



















































































































































    













































    



































































































