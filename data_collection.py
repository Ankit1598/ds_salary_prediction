# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:21:47 2020

@author: Ankit Chaudhari
"""

import glassdoor_scraper as gs
import pandas as pd

path = "E:/Data Science Projects/ds_salary_prediction/chromedriver"

jobs = gs.get_jobs("Data Scientist", 2000, False, path, 10)
jobs.to_csv('glassdoor_ds_jobs.csv', index = False)