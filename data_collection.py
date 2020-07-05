# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:21:47 2020

@author: Ankit Chaudhari
"""

import glassdoor_scraper as gs
import pandas as pd

path = "E:/Data Science Projects/ds_salary_prediction/chromedriver"

jobs = gs.get_jobs("Data Scientist", 20, False, path, 10)