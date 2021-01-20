"""
Author: Sharath Chandra Nagulapally, Sarthak Gupte, Abhinandan Desai
"""
from sqlalchemy import create_engine
import io
import psycopg2 as psy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math

"""
Please install the above dependencies before you run the program.
"""

"""To display all columns of pandas dataframe"""
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


"""
This function is used for merging the 8 datasets (2011-2018) from https://data.boston.gov/dataset/employee-earnings-report
Pandas is used to make the column names consistent across all the datasets which makes it easier to merge the datasets.
'$' and ',' are replaced in the appropriate attributes. 

In this phase, Data Cleaning is done as well: All the columns are brought into unique format, '$',',' are removed from 
many datasets and also the datatype of the columns are changed to the write format.
"""


def loading_first_dataset():
    dataset2011 = pd.read_csv("employee-earnings-report-2011.csv")
    dataset2012 = pd.read_csv("employee-earnings-report-2012.csv")
    dataset2013 = pd.read_csv("employee-earnings-report-2013.csv")
    dataset2014 = pd.read_csv("employee-earnings-report-2014.csv")
    dataset2015 = pd.read_csv("employee-earnings-report-2015.csv")
    dataset2016 = pd.read_csv("employee-earnings-report-2016.csv", encoding='latin-1')
    dataset2017 = pd.read_csv("employee-earnings-report-2017.csv")
    dataset2018 = pd.read_csv("employeeearningscy18full.csv", encoding='latin-1')

    dataset2014.rename(columns={'DEPARTMENT NAME': 'DEPARTMENT'}, inplace=True)
    dataset2013And2014Merged = pd.concat([dataset2013, dataset2014])
    dataset2011.rename(columns={'Name': 'NAME', 'Department Name': 'DEPARTMENT', 'Title': 'TITLE',
                                'Regular': 'REGULAR', 'Retro': 'RETRO', 'Other': 'OTHER', 'Overtime': 'OVERTIME',
                                'Injured': 'INJURED', 'Detail': 'DETAIL', 'Quinn': 'QUINN',
                                'Total Earnings': 'TOTAL EARNINGS', 'Zip Code': 'ZIP'}, inplace=True)
    dataset2012.rename(columns={'DETAIL ': 'DETAIL'}, inplace=True)
    dataset2015.rename(columns={'DEPARTMENT_NAME': 'DEPARTMENT', 'DETAILS': 'DETAIL',
                                'QUINN/EDUCATION INCENTIVE': 'QUINN', 'POSTAL': 'ZIP'}, inplace=True)
    mergedDataset = pd.concat([dataset2011, dataset2012, dataset2015])
    dataset2016.rename(columns={'DEPARTMENT_NAME': 'DEPARTMENT'}, inplace=True)
    dataset2017.rename(columns={'DEPARTMENT NAME': 'DEPARTMENT'}, inplace=True)
    dataset2018.rename(columns={'DEPARTMENT_NAME': 'DEPARTMENT'}, inplace=True)
    mergedDataset2 = pd.concat([dataset2016, dataset2017, dataset2018])
    mergedDataset2.rename(columns={'QUINN/EDUCATION INCENTIVE': 'QUINN', 'POSTAL': 'ZIP'}, inplace=True)
    finalDataset = pd.concat([dataset2013And2014Merged, mergedDataset, mergedDataset2])
    finalDataset = finalDataset[['DEPARTMENT', 'NAME', 'TITLE', 'REGULAR', 'OVERTIME', 'TOTAL EARNINGS']]

    """Removing $ and comma from the the numeric columns so that they can be loaded to postgreSQL database"""
    finalDataset["REGULAR"] = finalDataset["REGULAR"].astype(str)
    finalDataset.REGULAR = [x[1:] for x in finalDataset.REGULAR]
    finalDataset['REGULAR'] = finalDataset['REGULAR'].str.replace(',', '')
    finalDataset["OVERTIME"] = finalDataset["OVERTIME"].astype(str)
    finalDataset.OVERTIME = [x[1:] for x in finalDataset.OVERTIME]
    finalDataset['OVERTIME'] = finalDataset['OVERTIME'].str.replace(',', '')
    finalDataset["TOTAL EARNINGS"] = finalDataset["TOTAL EARNINGS"].astype(str)
    finalDataset['TOTAL EARNINGS'] = [x[1:] for x in finalDataset['TOTAL EARNINGS']]
    finalDataset['TOTAL EARNINGS'] = finalDataset['TOTAL EARNINGS'].str.replace(',', '')
    return finalDataset


""" This function loads data from https://data.ny.gov/Transparency/Salary-Information-for-State-Authorities/unag-2p27 
    which is the second source. Pandas is used to make the column names consistent across all the datasets 
    which makes it easier to merge the datasets."""


def loading_second_dataset():
    dataset = pd.read_csv("Salary_Information_for_State_Authorities.csv", low_memory=False)
    dataset['Name'] = dataset['First Name'] + ' ' + dataset['Last Name']
    dataset = dataset[['Authority Name', 'Name', 'Title', 'Base Annualized Salary',
                       'Overtime Paid', 'Total Compensation']]
    dataset.rename(columns={'Authority Name': 'DEPARTMENT', 'Name': 'NAME', 'Title': 'TITLE',
                            'Base Annualized Salary': 'REGULAR', 'Overtime Paid': 'OVERTIME',
                            'Total Compensation': 'TOTAL EARNINGS'}, inplace=True)
    return dataset


"""Just merging the two datasets which came from different sources and also changing the datatype to float for 
three attributes - regular pay, overtime pay, total earnings. In this function, data cleaning is done as well - null 
values are removed which provide incorrect values for descriptive statistics."""


def loading_datasets():
    first_dataset = loading_first_dataset()
    second_dataset = loading_second_dataset()
    dataset = pd.concat([first_dataset, second_dataset])
    dataset['REGULAR'] = pd.to_numeric(dataset['REGULAR'], errors='coerce', downcast='float')
    dataset['OVERTIME'] = pd.to_numeric(dataset['OVERTIME'], errors='coerce', downcast='float')
    dataset['TOTAL EARNINGS'] = pd.to_numeric(dataset['TOTAL EARNINGS'], errors='coerce', downcast='float')
    dataset['REGULAR'] = dataset['REGULAR'].clip(lower=0)
    dataset['OVERTIME'] = dataset['OVERTIME'].clip(lower=0)
    dataset['TOTAL EARNINGS'] = dataset['TOTAL EARNINGS'].clip(lower=0)
    dataset['OVERTIME'] = dataset['OVERTIME'].replace(0, np.nan)
    dataset['REGULAR'] = dataset['REGULAR'].replace(0, np.nan)
    dataset['TOTAL EARNINGS'] = dataset['TOTAL EARNINGS'].replace(0, np.nan)
    print("Dataset from multiple sources are merged!")
    dataset['REGULAR'] = dataset['REGULAR'].round(decimals=2)
    dataset['OVERTIME'] = dataset['OVERTIME'].round(decimals=2)
    dataset['TOTAL EARNINGS'] = dataset['TOTAL EARNINGS'].round(decimals=2)
    """It is important to drop null values or else the results might be inaccurate"""
    dataset.dropna(inplace=True)
    describeArray = dataset.describe()

    print(describeArray)
    print("Interquartile Range (IQR) \n", describeArray.iloc[6] - describeArray.iloc[4])
    print("Median: Regular", dataset["REGULAR"].median())
    print("Median: OVERTIME", dataset["OVERTIME"].median())
    print("Median: TOTAL EARNINGS", dataset["TOTAL EARNINGS"].median())
    print("Mode: Regular", dataset["REGULAR"].mode())
    print("Mode: OVERTIME", dataset["OVERTIME"].mode())
    print("Mode: TOTAL EARNINGS", dataset["TOTAL EARNINGS"].mode())
    print()
    print()
    return dataset


def rawDataset():
    first_dataset = loading_first_dataset()
    second_dataset = loading_second_dataset()
    raw_dataset = pd.concat([first_dataset, second_dataset])
    raw_dataset['REGULAR'] = pd.to_numeric(raw_dataset['REGULAR'], errors='coerce', downcast='float')
    raw_dataset['OVERTIME'] = pd.to_numeric(raw_dataset['OVERTIME'], errors='coerce', downcast='float')
    raw_dataset['TOTAL EARNINGS'] = pd.to_numeric(raw_dataset['TOTAL EARNINGS'], errors='coerce', downcast='float')
    print("Dataset from multiple sources are merged!")

    """It is important to drop null values or else the results might be inaccurate"""
    raw_dataset.dropna(inplace=True)
    return raw_dataset


"""Using sqlalchemy to connect with postgreSQL database and wire the dataframe into table"""


def dataframe_to_database(dataset):
    # DATABASE_URI = 'postgres+psycopg2://username:password@localhost:5432/databasename' this is an example, you need to
    # replace the username, password, databasename values with your credentials.
    DATABASE_URI = 'postgres+psycopg2://postgres:1234@localhost:5432/DataCleaning'
    engine = create_engine(DATABASE_URI)
    dataset.head(0).to_sql('merged_dataset', engine, if_exists='replace', index=False)  # truncates the table
    conn = engine.raw_connection()
    cur = conn.cursor()

    output = io.StringIO()
    dataset.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    cur.copy_from(output, 'merged_dataset', null="")  # null values become ''
    #  engine.execute('alter table merged_dataset add primary key(row_num)')
    conn.commit()
    print()
    print("Copy from DataFrame to postgreSQL is done!")


"""This function connects with postgreSQL using psycopg2 driver and add a primary key ID which is 
generated incrementally"""


def adding_primarykey():
    cursor = ""
    connection = ""
    try:
        connection = psy.connect(user="postgres",
                                 password="1234",
                                 host="localhost",
                                 port="5432",
                                 database="DataCleaning")
        cursor = connection.cursor()
        cursor.execute('ALTER TABLE merged_dataset ADD COLUMN ID SERIAL PRIMARY KEY;')
        print("Primary Key added!")
        connection.commit()
    except(Exception, psy.DatabaseError) as e:
        print(e)
    finally:
        cursor.close()
        connection.close()

"""
This function is used for calculating means of the raw data and clean data for each department which is further used in 
visualization.
"""


def calculateMean(dataset):
    temp_list = list(dataset["DEPARTMENT"].unique())
    mean_list = []
    for each in temp_list:
        temp = dataset.loc[dataset['DEPARTMENT'] == each]
        mean_list.append(temp['TOTAL EARNINGS'].mean())
    #print(mean_list)
    return temp_list, mean_list


"""
Bar graph for Raw Data vs Clean Data
"""


def visualization(rawData, cleanData):
    # print("Raw ", rawData.head())
    # print("Clean ", cleanData.head())
    column_list, mean_list = calculateMean(rawData)
    columnList, meansList = calculateMean(cleanData)
    difference = []
    zip_object = zip(meansList, mean_list)
    for list1_i, list2_i in zip_object:
        difference.append(list1_i - list2_i)
    #print(column_list)
    #print("the difference of clean-raw ", difference)
    columnsFinalList = [column_list[0], column_list[48], column_list[103]]
    rawDataFinalList = [mean_list[0], mean_list[48], mean_list[103]]
    cleanDataFinalList = [meansList[0], meansList[48], meansList[103]]
    labels = columnsFinalList
    x = np.arange(len(columnsFinalList))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, rawDataFinalList, width, label='Raw Data')
    ax.bar(x + width / 2, cleanDataFinalList, width, label='Clean Data')
    ax.set_xlabel('Department Name')
    ax.set_ylabel('Average Mean of Total Earnings')
    ax.set_title('Average Income for Departments')
    ax.set_xticks(x)
    ax.set_xticklabels(columnsFinalList)
    ax.legend()
    fig.tight_layout()
    plt.savefig("Raw Data vs Clean Data.png")
    plt.show()


def main():
    dataset = loading_datasets()
    columns_list = ['REGULAR', 'OVERTIME', 'TOTAL EARNINGS']
    raw_dataset = rawDataset()
    dataframe_to_database(dataset)
    adding_primarykey()
    dataset = addingTargetClass(dataset)
    print("Dataset head")
    print(dataset.head())
    print()

    print("Class frequency")
    print(dataset["TARGET"].value_counts())
    print()
    data_mining(dataset)
    visualization(raw_dataset, dataset)
    dataset = dataset[['REGULAR', 'OVERTIME', 'TARGET']]
    dataset, values_list = loading_data(dataset, columns_list)
    #dec_data = dataset.copy()
    #decimal(dec_data, columns_list)
    std_data = dataset.copy()
    standard(std_data, values_list, columns_list)
    # minMaxData = dataset.copy()
    # min_max(minMaxData, columns_list)
    sigmoidal_data = dataset.copy()
    sigmoidal(sigmoidal_data, values_list, columns_list)
    #softmax_data = dataset.copy()
    #softmax(softmax_data, values_list, columns_list)
    print()
    print()

"""
This function adds the target class to the dataset.
"""


def addingTargetClass(dataset):
    dataset['TARGET'] = dataset.apply(s, axis=1)
    return dataset


def s(row):
    if row['TOTAL EARNINGS'] < 35000:
        val = 0
    elif row['TOTAL EARNINGS'] < 70000:
        val = 1
    elif row['TOTAL EARNINGS'] < 105000:
        val = 2
    else:
        val = 3
    return val


def standard(data, values_list, columns_list):
    for i in range(2):
        data[columns_list[i]] = data[columns_list[i]].apply(lambda x: ((x-values_list[i][0])/values_list[i][1]))
    print("Standard Normalization ", end="")
    #print(data.head())
    data_mining(data)
    return data


def min_max(data, columns_list):
    for i in range(2):
        minValue = data[columns_list[i]].min()
        maxValue = data[columns_list[i]].max()
        data[columns_list[i]] = data[columns_list[i]].apply(lambda x: ((x - minValue) /(maxValue-minValue)))
    print("Min-max normalization ", end="")
    #print(data)
    data_mining(data)
    return data


def decimal(data, columns_list):
    for i in range(2):
        max_value = len(str(int(max(data[columns_list[i]]))))
        data[columns_list[i]] = data[columns_list[i]].apply(lambda x: (x/10**max_value))
    print("Decimal normalization ", end="")
    data_mining(data)
    pass

"""
The formula for sigmoidal function is first calculate a =((x-mean)/std dev) and then x = (1-e**(-a))/(1+e**(a)) 
e**a =  e power(a)  
"""


def sigmoidal(data, values_list, columns_list):
    for i in range(2):
        data[columns_list[i]] = data[columns_list[i]].apply(lambda x: ((x - values_list[i][0]) / values_list[i][1]))
        data[columns_list[i]] = data[columns_list[i]].apply(lambda x: ((1-(math.exp(-x))) /(1+(math.exp(x)))))
    print("Sigmoidal Normalization ", end="")
    data_mining(data)
    return data

"""
The formula for softmax function is first calculate a =((x-mean)/std dev) and then x = (1/(1+e**(a)) combineClasses(temp_data, values_list, columns_list)alue
e**a =  e power(a)  
"""


def softmax(data, values_list, columns_list):
    for i in range(2):
        data[columns_list[i]] = data[columns_list[i]].apply(lambda x: ((x - values_list[i][0]) / values_list[i][1]))
        data[columns_list[i]] = data[columns_list[i]].apply(lambda x: (1 / (1+(math.exp(x)))))
    print("Softmax normalization ", end="")
    data_mining(data)
    return data


def loading_data(data, columns_list):
    values_list = []
    for i in range(2):
        data[columns_list[i]] = data[columns_list[i]].astype(float)
        values_list.append([data.loc[:, columns_list[i]].mean(), data.loc[:, columns_list[i]].std()])
    print("Unnormalized ", end="")
    data_mining(data)
    return data, values_list

"""
Decision trees classifier is used in this function.
"""


def data_mining(data):
    data = data[['REGULAR', 'OVERTIME', 'TARGET']]
    X = data.iloc[:, 0:2].values
    y = data.iloc[:, 2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("accuracy is ", (accuracy_score(y_test, y_pred)*100).round(decimals=2))
    print("accuracy of zero classifier is ", str(round((zero_classifier(y_test)*100), 2)))
    print()


def zero_classifier(y_test):
    return (list(y_test).count(2))/len(list(y_test))


if __name__ == '__main__':
    main()