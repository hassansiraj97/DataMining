# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:33:02 2024

@author: Hassan
"""

from ucimlrepo import fetch_ucirepo 
import sweetviz as sv
  
# fetch dataset 
regensburg_pediatric_appendicitis = fetch_ucirepo(id=938) 
  
# data (as pandas dataframes) 
X = regensburg_pediatric_appendicitis.data.features 
y = regensburg_pediatric_appendicitis.data.targets 
  
# metadata 
print(regensburg_pediatric_appendicitis.metadata) 
  
# variable information 
print(regensburg_pediatric_appendicitis.variables) 

my_report = sv.analyze(X)
my_report.show_html() # This will generate an HTML report
