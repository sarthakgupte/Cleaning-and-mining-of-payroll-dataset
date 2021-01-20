# Cleaning-and-mining-of-payroll-dataset

Installation Guide

The following dependencies need to be installed before you run the program:


1. Pandas: pandas can be installed by executing "pip install pandas" in the terminal.
2. Numpy: this comes with pandas.
3. Sqlalchemy: this can be installed by executing "pip install sqlalchemy" in the terminal
4. Psycopg2: this can be installed by executing "pip install psycopg2" in the terminal.

Dataset Links: 
    - Analyze Boston. Employee Earnings Report.
        https://data.boston.gov/dataset/employee-earnings-report
    -  NY Open Data. Salary Information for StateAuthorities.
        https://data.ny.gov/Transparency/Salary-Information-for-State-Authorities/unag-2p27


After installing the above dependencies, you need to create a Database in postgreSQL database, this can be easily done by installing "PGAdmin" executable file, 
and then you could create the database from there. You can create a new user or use the default "postgres" user. When prompted enter the password you want - all these 
credentials will need to be replaced in the "dataframe_to_database" function and the "adding_primarykey" function as your localhost credentials will be different than mine. 
And please make sure the program code and the datasets are in the same directory.
