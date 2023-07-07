import sqlite3
import pandas as pd


# read data
data=pd.read_csv('all_skills_list.csv')#job_data_with_skills_v1.csv



####################################################################################################

# Create a SQLite database and insert job data
conn = sqlite3.connect('all_skills.db')
c = conn.cursor()
c.execute('''CREATE TABLE jobs (skill_name TEXT, skill_type TEXT)''')

for idx, row in data.iterrows():
    skill_name = row['name']
    skill_type = row['type.name']
    c.execute('''INSERT INTO jobs (skill_name, skill_type) VALUES (?, ?)''', (skill_name, skill_type))

conn.commit()
conn.close()