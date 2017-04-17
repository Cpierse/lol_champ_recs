# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:44:20 2016
Takes our JSON file and writes the content to an sql database

@author: Chris Pierse
"""
import sqlite3, json


#%% Load the data and initialize sql:

# Load the json:
with open('SummData.json', 'r') as fp:
    summ_data = json.load(fp)

# Initalize SQL:
con = sqlite3.connect('SummData.db')
cur = con.cursor()    

# Make the table:
#cur.execute("DROP TABLE SummData")
cur.execute("CREATE TABLE IF NOT EXISTS \
    SummData(Id INT PRIMARY KEY, \
    Tier INT, \
    Top VARCHAR, \
    Jungle VARCHAR, \
    Mid VARCHAR, \
    Support VARCHAR, \
    ADC VARCHAR)")

#%% Write the current data to the sql file:
for tier in summ_data:
    if "GOLD" in tier: tid = 2 # Only have data from GOLD right now.
    for sid in summ_data[tier]:
        cur.execute("INSERT INTO SummData VALUES({0},{1},'{2}','{3}','{4}','{5}','{6}')".format(
            str(sid),
            str(tid),
            str(summ_data[tier][sid].get("TOP","")),
            str(summ_data[tier][sid].get("JUNGLE","")),
            str(summ_data[tier][sid].get("MID","")),
            str(summ_data[tier][sid].get("SUPPORT","")),
            str(summ_data[tier][sid].get("ADC",""))
            ))


con.commit()
con.close()


