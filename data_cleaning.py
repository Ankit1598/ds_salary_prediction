# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:06:11 2020

@author: Ankit Chaudhari
"""

import pandas as pd

df = pd.read_csv("glassdoor_ds_jobs.csv")

# Salary Parsing
df = df[df['Salary Estimate'] != '-1']

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
min_kd = salary.apply(lambda x: x.replace('$', '').replace('K', ''))
min_hr = min_kd.apply(lambda x: x.lower().replace('per hour', ''))

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1].split()[0]))
df['avg_salary'] = (df.min_salary + df.max_salary) / 2

# Company Name
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] == -1 else x['Company Name'][:-3], axis = 1)

#State Field
df['job_state'] = df['Location'].apply(lambda x: x.split(', ')[1] if ',' in x else x)
df.job_state = df.job_state.replace({'Utah':'UT',
									 'United States': 'US',
									 'New Jersey': 'NJ',
									 'Illinois':'IL',
									 'Remote': 'Remote'})

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

# Company Age
df['comp_age'] = df.Founded.apply(lambda x: x if x == -1 else 2020-x)

# Parsing Job Descriptions
#Py
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df.python_yn.value_counts()
#R
df['r_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df.r_yn.value_counts()
#Spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark_yn.value_counts()
#AWS
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws_yn.value_counts()
#Excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel_yn.value_counts()

df.to_csv('salary_data_cleaned.csv', index = False)
