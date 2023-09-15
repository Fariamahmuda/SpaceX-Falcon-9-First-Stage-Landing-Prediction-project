#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Requests allows us to make HTTP requests which we will use to get data from an API
import requests
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Datetime is a library that allows us to represent dates
import datetime

# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns', None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)


# In[2]:


import sys
from bs4 import BeautifulSoup
import re
import unicodedata
import plotly.express as px
import sys
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


# Takes the dataset and uses the rocket column to call the API and append the data to the list
def getBoosterVersion(data):
    for x in data['rocket']:
       if x:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])


# In[5]:


# Takes the dataset and uses the launchpad column to call the API and append the data to the list
def getLaunchSite(data):
    for x in data['launchpad']:
       if x:
         response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
         Longitude.append(response['longitude'])
         Latitude.append(response['latitude'])
         LaunchSite.append(response['name'])


# In[6]:


# Takes the dataset and uses the payloads column to call the API and append the data to the lists
def getPayloadData(data):
    for load in data['payloads']:
       if load:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])


# In[7]:


# Takes the dataset and uses the cores column to call the API and append the data to the lists
def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])


# In[8]:


spacex_url="https://api.spacexdata.com/v4/launches/past"


# In[9]:


response = requests.get(spacex_url)


# In[10]:


print(response.content)


# Task 1: Request and parse the SpaceX launch data using the GET request
# To make the requested JSON results more consistent, we will use the following static response object for this project:
# 
# 

# In[11]:


static_json_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'


# In[12]:


response.status_code


# In[13]:


# Use json_normalize method to convert the json result into a dataframe
data = pd.json_normalize(response.json())


# In[14]:


data.head(5)


# We will notice that a lot of the data are IDs. For example the rocket column has no information about the rocket just an identification number.
# 
# We will now use the API again to get information about the launches using the IDs given for each launch. Specifically we will be using columns rocket, payloads, launchpad, and cores.

# In[15]:


# Lets take a subset of our dataframe keeping only the features we want and the flight number, and date_utc.
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

# We will remove rows with multiple cores because those are falcon rockets with 2 extra rocket boosters and rows that have multiple payloads in a single rocket.
data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]

# Since payloads and cores are lists of size 1 we will also extract the single value in the list and replace the feature.
data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])

# We also want to convert the date_utc to a datetime datatype and then extracting the date leaving the time
data['date'] = pd.to_datetime(data['date_utc']).dt.date

# Using the date we will restrict the dates of the launches
data = data[data['date'] <= datetime.date(2020, 11, 13)]


# From the rocket we would like to learn the booster name
# 
# From the payload we would like to learn the mass of the payload and the orbit that it is going to
# 
# From the launchpad we would like to know the name of the launch site being used, the longitude, and the latitude.
# 
# From cores we would like to learn the outcome of the landing, the type of the landing, number of flights with that core, whether gridfins were used, whether the core is reused, whether legs were used, the landing pad used, the block of the core which is a number used to seperate version of cores, the number of times this specific core has been reused, and the serial of the core.
# 
# The data from these requests will be stored in lists and will be used to create a new dataframe.
# 
# 

# In[16]:


#Global variables 
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []


# These functions will apply the outputs globally to the above variables. Let's take a looks at BoosterVersion variable. Before we apply getBoosterVersion the list is empty:

# In[17]:


BoosterVersion
[]


# Now, let's apply  getBoosterVersion function method to get the booster version

# In[18]:


# Call getBoosterVersion
getBoosterVersion(data)
#the list has now been update


# In[19]:


BoosterVersion[0:5]
['Falcon 1', 'Falcon 1', 'Falcon 1', 'Falcon 1', 'Falcon 9']
#we can apply the rest of the functions here:


# In[20]:


# Call getLaunchSite
getLaunchSite(data)


# In[21]:


# Call getPayloadData
getPayloadData(data)


# In[22]:


# Call getCoreData
getCoreData(data)


# Finally lets construct our dataset using the data we have obtained. We we combine the columns into a dictionary.

# In[23]:


launch_dict = {'FlightNumber': list(data['flight_number']),
'Date': list(data['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}


# Then, we need to create a Pandas data frame from the dictionary launch_dict.

# In[24]:


data_falcon9 = pd.DataFrame(launch_dict)
print(data_falcon9.head())


# In[25]:


# Create a data from launch_dict
df = pd.DataFrame(launch_dict)


# Show the summary of the dataframe
# 

# In[26]:


# Show the head of the dataframe
df.head()


# Task 2: Filter the dataframe to only include Falcon 9 launches
# Finally we will remove the Falcon 1 launches keeping only the Falcon 9 launches. Filter the data dataframe using the BoosterVersion column to only keep the Falcon 9 launches. Save the filtered data to a new dataframe called data_falcon9.

# In[27]:


# Hint data['BoosterVersion']!='Falcon 1'
data_falcon9 = df[df['BoosterVersion'] != 'Falcon 1']


# 
# Now that we have removed some values we should reset the FlgihtNumber column
# 
# 

# In[28]:


data_falcon9.loc[:,'FlightNumber'] = list(range(1, data_falcon9.shape[0]+1))
data_falcon9

Data Wrangling
# We can see below that some of the rows are missing values in our dataset.

# In[29]:


data_falcon9.isnull().sum()


# Before we can continue we must deal with these missing values. The LandingPad column will retain None values to represent when landing pads were not used.

# Task 3: Dealing with Missing Values
# Calculate below the mean for the PayloadMass using the .mean(). Then use the mean and the .replace() function to replace np.nan values in the data with the mean you calculated.

# In[30]:


# Calculate the mean value of PayloadMass column

payload_mean = df['PayloadMass'].mean()

# Replace the np.nan values with its mean value

df['PayloadMass'].replace(np.nan, payload_mean, inplace=True)


# In[31]:


data_falcon9.isnull().sum()


# You should see the number of missing values of the PayLoadMass change to zero.
# 
# Now we should have no missing values in our dataset except for in LandingPad.
# 
# We can now export it to a CSV for the next section,but to make the answers consistent, in the next lab we will provide data in a pre-selected date range.
# 
# 

# In[32]:


data_falcon9.to_csv('dataset_part_1.csv', index=False)


# In[33]:


get_ipython().system('pip3 install beautifulsoup4')
get_ipython().system('pip3 install requests')


# and we will provide some helper functions to process web scraped HTML table

# In[34]:


def date_time(table_cells):
    """
    This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """
    This function returns the booster version from the HTML  table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=[i for i in table_cells.strings][0]
    return out
def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass


def extract_column_from_header(row):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    


# To keep the lab tasks consistent, you will be asked to scrape the data from a snapshot of the List of Falcon 9 and Falcon Heavy launches Wikipage updated on 9th June 2021

# In[35]:


static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"


# Next, request the HTML page from the above URL and get a response object
TASK 1: Request the Falcon9 Launch Wiki page from its URL
# First, let's perform an HTTP GET method to request the Falcon9 Launch HTML page, as an HTTP response.

# In[36]:


# use requests.get() method with the provided static_url
# assign the response to a object


# In[37]:


r = requests.get(static_url)
data = r.text


# Create a BeautifulSoup object from the HTML response

# In[38]:


# Use BeautifulSoup() to create a BeautifulSoup object from a response text content
soup = BeautifulSoup(data,"html.parser")


# Print the page title to verify if the BeautifulSoup object was created properly

# In[39]:


# Use soup.title attribute
print(soup.title)


# # TASK 2: Extract all column/variable names from the HTML table header
# Next, we want to collect all relevant column names from the HTML table header
# 
# Let's try to find all tables on the wiki page first. If we need to refresh our memory about BeautifulSoup,we will check the external reference link towards the end of this lab

# In[40]:


# Use the find_all function in the BeautifulSoup object, with element type `table`
# Assign the result to a list called `html_tables`


# In[41]:


html_tables = soup.find_all('table')


# In[42]:


# Let's print the third table and check its content
first_launch_table = html_tables[2]
print(first_launch_table)

Next, we just need to iterate through the <th> elements and apply the provided extract_column_from_header() to extract column name one by one
# In[43]:


column_names = []

# Apply find_all() function with `th` element on first_launch_table
# Iterate each th element and apply the provided extract_column_from_header() to get a column name
# Append the Non-empty column name (`if name is not None and len(name) > 0`) into a list called column_names
table_headers = first_launch_table.find_all('th')
# print(table_headers)
for j, table_header in enumerate(table_headers):
    name = extract_column_from_header(table_header)
    if name is not None and len(name) > 0:
        column_names.append(name)
    
print(column_names)


# # TASK 3: Create a data frame by parsing the launch HTML tables
We will create an empty dictionary with keys from the extracted column names in the previous task. Later, this dictionary will be converted into a Pandas dataframe
# In[44]:


launch_dict= dict.fromkeys(column_names)

# Remove an irrelvant column
del launch_dict['Date and time ( )']

# Let's initial the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]

Next, we just need to fill up the launch_dict with launch records extracted from table rows.

Usually, HTML tables in Wiki pages are likely to contain unexpected annotations and other types of noises, such as reference links B0004.1[8], missing values N/A [e], inconsistent formatting, etc.

To simplify the parsing process, we have provided an incomplete code snippet below to help to fill up the launch_dict. We have to complete the following code snippet with TODOs or we can choose to write our own logic to parse all launch tables:
# In[45]:


extracted_row = 0
#Extract each table 
for table_number,table in enumerate(soup.find_all('table',"wikitable plainrowheaders collapsible")):
   # get table row 
    for rows in table.find_all("tr"):
        #check to see if first table heading is as number corresponding to launch a number 
        if rows.th:
            if rows.th.string:
                flight_number=rows.th.string.strip()
                flag=flight_number.isdigit()
        else:
            flag=False
        #get table element 
        row=rows.find_all('td')
        #if it is number save cells in a dictonary 
        if flag:
            extracted_row += 1
            # Flight Number value
            # TODO: Append the flight_number into launch_dict with key `Flight No.`
            #print(flight_number)
            datatimelist=date_time(row[0])
            
            # Date value
            # TODO: Append the date into launch_dict with key `Date`
            date = datatimelist[0].strip(',')
            #print(date)
            
            # Time value
            # TODO: Append the time into launch_dict with key `Time`
            time = datatimelist[1]
            #print(time)
              
            # Booster version
            # TODO: Append the bv into launch_dict with key `Version Booster`
            bv=booster_version(row[1])
            if not(bv):
                bv=row[1].a.string
            print(bv)
            
            # Launch Site
            # TODO: Append the bv into launch_dict with key `Launch Site`
            launch_site = row[2].a.string
            #print(launch_site)
            
            # Payload
            # TODO: Append the payload into launch_dict with key `Payload`
            payload = row[3].a.string
            #print(payload)
            
            # Payload Mass
            # TODO: Append the payload_mass into launch_dict with key `Payload mass`
            payload_mass = get_mass(row[4])
            #print(payload)
            
            # Orbit
            # TODO: Append the orbit into launch_dict with key `Orbit`
            orbit = row[5].a.string
            #print(orbit)
            
            # Customer
            # TODO: Append the customer into launch_dict with key `Customer`
            #customer = row[6].a.string
            #print(customer)
            try:
                customer = row[6].a.string
            except:
                customer = "None"
            
            # Launch outcome
            # TODO: Append the launch_outcome into launch_dict with key `Launch outcome`
            launch_outcome = list(row[7].strings)[0]
            #print(launch_outcome)
            
            # Booster landing
            # TODO: Append the launch_outcome into launch_dict with key `Booster landing`
            booster_landing = landing_status(row[8])
            print(booster_landing)
            
print("number of extracted rows: ", extracted_row) 


# After we have fill in the parsed launch record values into launch_dict, we can create a dataframe from it.

# In[46]:


df=pd.DataFrame(launch_dict)


# In[47]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np


# Data analysis

# In[48]:


df = data_falcon9


# In[49]:


df.head(10)


# Identify and calculate the percentage of the missing values in each attribute

# In[50]:


df.isnull().sum()/df.shape[0]*100


# In[51]:


df.dtypes


# # TASK 1: Calculate the number of launches on each site
The data contains several Space X launch facilities: Cape Canaveral Space Launch Complex 40 VAFB SLC 4E , Vandenberg Air Force Base Space Launch Complex 4E (SLC-4E), Kennedy Space Center Launch Complex 39A KSC LC 39A .The location of each Launch Is placed in the column LaunchSite

Next, let's see the number of launches for each site.

Use the method value_counts() on the column LaunchSite to determine the number of launches on each site:
# In[52]:


# Apply value_counts() on column LaunchSite
launch_site_counts = df['LaunchSite'].value_counts()
print(launch_site_counts)


# Each launch aims to an dedicated orbit, and here are some common orbit types:
# 
# LEO: Low Earth orbit (LEO)is an Earth-centred orbit with an altitude of 2,000 km (1,200 mi) or less (approximately one-third of the radius of Earth),[1] or with at least 11.25 periods per day (an orbital period of 128 minutes or less) and an eccentricity less than 0.25.[2] Most of the manmade objects in outer space are in LEO [1].
# 
# VLEO: Very Low Earth Orbits (VLEO) can be defined as the orbits with a mean altitude below 450 km. Operating in these orbits can provide a number of benefits to Earth observation spacecraft as the spacecraft operates closer to the observation[2].
# 
# GTO A geosynchronous orbit is a high Earth orbit that allows satellites to match Earth's rotation. Located at 22,236 miles (35,786 kilometers) above Earth's equator, this position is a valuable spot for monitoring weather, communications and surveillance. Because the satellite orbits at the same speed that the Earth is turning, the satellite seems to stay in place over a single longitude, though it may drift north to south,” NASA wrote on its Earth Observatory website [3] .
# 
# SSO (or SO): It is a Sun-synchronous orbit also called a heliosynchronous orbit is a nearly polar orbit around a planet, in which the satellite passes over any given point of the planet's surface at the same local mean solar time [4] .
# 
# ES-L1 :At the Lagrange points the gravitational forces of the two large bodies cancel out in such a way that a small object placed in orbit there is in equilibrium relative to the center of mass of the large bodies. L1 is one such point between the sun and the earth [5] .
# 
# HEO A highly elliptical orbit, is an elliptic orbit with high eccentricity, usually referring to one around Earth [6].
# 
# ISS A modular space station (habitable artificial satellite) in low Earth orbit. It is a multinational collaborative project between five participating space agencies: NASA (United States), Roscosmos (Russia), JAXA (Japan), ESA (Europe), and CSA (Canada) [7]
# 
# MEO Geocentric orbits ranging in altitude from 2,000 km (1,200 mi) to just below geosynchronous orbit at 35,786 kilometers (22,236 mi). Also known as an intermediate circular orbit. These are "most commonly at 20,200 kilometers (12,600 mi), or 20,650 kilometers (12,830 mi), with an orbital period of 12 hours [8]
# 
# HEO Geocentric orbits above the altitude of geosynchronous orbit (35,786 km or 22,236 mi) [9]
# 
# GEO It is a circular geosynchronous orbit 35,786 kilometres (22,236 miles) above Earth's equator and following the direction of Earth's rotation [10]
# 
# PO It is one type of satellites in which a satellite passes above or nearly above both poles of the body being orbited (usually a planet such as the Earth [11]
# 
# some are shown in the following plot:
# 

# ![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/Orbits.png)
# 

# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/Orbits.png

# # TASK 2: Calculate the number and occurrence of each orbit
Use the method .value_counts() to determine the number and occurrence of each orbit in the column Orbit
# In[53]:


# Apply value_counts on Orbit column
orbit_counts = df['Orbit'].value_counts()
print(orbit_counts)


# # TASK 3: Calculate the number and occurence of mission outcome per orbit type

# Use the method .value_counts() on the column Outcome to determine the number of landing_outcomes.Then assign it to a variable landing_outcomes.

# In[54]:


# landing_outcomes = values on Outcome column
landing_outcomes = df['Outcome'].value_counts()
print(landing_outcomes)


# True Ocean means the mission outcome was successfully landed to a specific region of the ocean while False Ocean means the mission outcome was unsuccessfully landed to a specific region of the ocean. True RTLS means the mission outcome was successfully landed to a ground pad False RTLS means the mission outcome was unsuccessfully landed to a ground pad.True ASDS means the mission outcome was successfully landed to a drone ship False ASDS means the mission outcome was unsuccessfully landed to a drone ship. None ASDS and None None these represent a failure to land.

# In[55]:


for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)


# We create a set of outcomes where the second stage did not land successfully:

# In[56]:


bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes


# # TASK 4: Create a landing outcome label from Outcome column

# Using the Outcome, create a list where the element is zero if the corresponding row in Outcome is in the set bad_outcome; otherwise, it's one. Then assign it to the variable landing_class:

# In[57]:


# landing_class = 0 if bad_outcome
# landing_class = 1 otherwise

bad_outcome = {'failure', 'lost'}
landing_class = [0 if outcome in bad_outcome else 1 for outcome in df['Outcome']]


# This variable will represent the classification variable that represents the outcome of each launch. If the value is zero, the first stage did not land successfully; one means the first stage landed Successfully

# In[58]:


df['Class']=landing_class
df[['Class']].head(8)


# In[59]:


df.head(5)


# In[60]:


df["Class"].mean()


# In[61]:


df=pd.DataFrame(launch_dict)


# In[62]:


df = pd.DataFrame(launch_dict)
df.to_csv('spacex_web_scraped.csv', index=False)


# In[63]:


df.to_csv("dataset_part_2.csv", index=False)


# Lab 4: Analysis with SQL
# Note: the original IBM Lab would use IBM Watson with DB2 as database. We will use SQLite3 instead.
# 
# The original DB2 instruction can be viewed here (requires login)

# In[64]:


get_ipython().system('pip install sqlalchemy==1.3.9')


# # Connect to the database

# 
# Let us first load the SQL extension and establish a connection with the database

# In[65]:


import csv, sqlite3
print(sqlite3.version)
print(sqlite3.sqlite_version)
con = sqlite3.connect("my_data1.db")
cur = con.cursor()


# In[66]:


df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")


# # Tasks

# Now write and execute SQL queries to solve the assignment tasks.

# Note: If the column names are in mixed case enclose it in double quotes For Example "Landing_Outcome"

# # Task 1

# 
# Display the names of the unique launch sites in the space mission

# In[67]:


# Read the CSV file into a DataFrame
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Establish a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Save the DataFrame as an SQL table
df.to_sql("SPACEXTBL", conn, if_exists='replace', index=False)

# Execute the SQL query to display the names of the unique launch sites
query = """
SELECT DISTINCT "Launch_Site"
FROM SPACEXTBL;
"""
result = conn.execute(query).fetchall()

# Extract the launch site names from the result
launch_sites = [row[0] for row in result]

# Print the launch site names
print(launch_sites)


# # Task 2

# 
# Display 5 records where launch sites begin with the string 'CCA'

# In[68]:


# Execute the SQL query to display 5 records where launch sites begin with 'CCA'
query = """
SELECT *
FROM SPACEXTBL
WHERE "Launch_Site" LIKE 'CCA%'
LIMIT 5;
"""
result = conn.execute(query).fetchall()

# Print the query result
for row in result:
    print(row)


# # Task 3

# 
# Display the total payload mass carried by boosters launched by NASA (CRS)

# In[69]:


# Execute the SQL query to calculate the total payload mass for NASA (CRS) launches
query = """
SELECT SUM(payload_mass__kg_)
FROM SPACEXTBL
WHERE customer = 'NASA (CRS)';
"""
result = conn.execute(query).fetchone()

# Extract the total payload mass from the result
total_payload_mass = result[0]

# Print the total payload mass
print(total_payload_mass)


# # Task 4

# 
# Display average payload mass carried by booster version F9 v1.

# In[70]:


# Execute the SQL query to calculate the average payload mass for booster version F9 v1.1
query = """
SELECT AVG(payload_mass__kg_)
FROM SPACEXTBL
WHERE booster_version = 'F9 v1.1';
"""
result = conn.execute(query).fetchone()

# Extract the average payload mass from the result
average_payload_mass = result[0]

# Print the average payload mass
print(average_payload_mass)


# # Task 5

# 
# List the date when the first succesful landing outcome in ground pad was acheived.
# 

# Hint:Use min function

# In[71]:


# Execute the SQL query to find the date of the first successful landing outcome on a ground pad
query = """
SELECT MIN(date)
FROM SPACEXTBL
WHERE landing_outcome = 'Success (ground pad)';
"""
result = conn.execute(query).fetchone()

# Extract the date of the first successful landing outcome from the result
first_successful_landing_date = result[0]

# Print the date of the first successful landing outcome
print(first_successful_landing_date)


# # Task 6

# 
# List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000

# In[72]:


# Execute the SQL query to find the names of boosters with success in drone ship landing and payload mass between 4000 and 6000
query = """
SELECT booster_version
FROM SPACEXTBL
WHERE landing_outcome = 'Success (drone ship)'
AND payload_mass__kg_ > 4000
AND payload_mass__kg_ < 6000;
"""
result = conn.execute(query).fetchall()

# Extract the booster names from the result
booster_names = [row[0] for row in result]

# Print the names of the boosters
print(booster_names)


# # Task 7

# List the total number of successful and failure mission outcomes

# In[73]:


# Execute the SQL query to calculate the total number of successful and failure mission outcomes
query = """
SELECT landing_outcome, COUNT(*)
FROM SPACEXTBL
GROUP BY landing_outcome;
"""
result = conn.execute(query).fetchall()

# Print the total number of successful and failure mission outcomes
for row in result:
    outcome = row[0]
    count = row[1]
    print(f"{outcome}: {count}")


# # Task 8

# List the names of the booster_versions which have carried the maximum payload mass. Use a subquery

# In[74]:


# Execute the SQL query to find the names of booster versions with maximum payload mass
query = """
SELECT booster_version
FROM SPACEXTBL
WHERE payload_mass__kg_ = (
    SELECT MAX(payload_mass__kg_)
    FROM SPACEXTBL
);
"""
result = conn.execute(query).fetchall()

# Extract the booster version names from the result
booster_versions = [row[0] for row in result]

# Print the names of the booster versions
print(booster_versions)


# # Task 9

# List the records which will display the month names, failure landing_outcomes in drone ship ,booster versions, launch_site for the months in year 2015.
# 
# Note: SQLLite does not support monthnames. So we need to use substr(Date, 4, 2) as month to get the months and substr(Date,7,4)='2015' for year.

# In[75]:


# Execute the SQL query to retrieve the desired records
query = """
SELECT 
    CASE 
        WHEN SUBSTR(date, 4, 2) = '01' THEN 'January'
        WHEN SUBSTR(date, 4, 2) = '02' THEN 'February'
        WHEN SUBSTR(date, 4, 2) = '03' THEN 'March'
        WHEN SUBSTR(date, 4, 2) = '04' THEN 'April'
        WHEN SUBSTR(date, 4, 2) = '05' THEN 'May'
        WHEN SUBSTR(date, 4, 2) = '06' THEN 'June'
        WHEN SUBSTR(date, 4, 2) = '07' THEN 'July'
        WHEN SUBSTR(date, 4, 2) = '08' THEN 'August'
        WHEN SUBSTR(date, 4, 2) = '09' THEN 'September'
        WHEN SUBSTR(date, 4, 2) = '10' THEN 'October'
        WHEN SUBSTR(date, 4, 2) = '11' THEN 'November'
        WHEN SUBSTR(date, 4, 2) = '12' THEN 'December'
    END AS month,
    landing_outcome,
    booster_version,
    launch_site
FROM SPACEXTBL
WHERE SUBSTR(date, 7, 4) = '2015'
    AND landing_outcome LIKE 'Failure%'
    AND landing_outcome LIKE '%drone ship%';
"""
result = conn.execute(query).fetchall()

# Print the query result
for row in result:
    print(row)


# # Task 10

# 
# Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.

# In[76]:


# Execute the SQL query to rank the count of landing outcomes
query = """
SELECT Landing_Outcome, COUNT(*) as count
FROM SPACEXTBL
WHERE date BETWEEN '2010-06-04' AND '2017-03-20'
GROUP BY Landing_Outcome
ORDER BY count DESC;
"""
result = conn.execute(query).fetchall()

# Print the ranked count of landing outcomes
for row in result:
    outcome = row[0]
    count = row[1]
    print(f"{outcome}: {count}")


# # Exploratory Data Analysis
# 

# In[77]:


df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")


# In[78]:


# Create a DataFrame from launch_dict
df = pd.DataFrame(launch_dict)

# Save the DataFrame to 'spacex_web_scraped.csv' without overwriting df
df.to_csv('spacex_web_scraped.csv', index=False)

# Save the DataFrame to 'dataset_part_2.csv'
df.to_csv("dataset_part_2.csv", index=False)


# In[79]:


URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"

df = pd.read_csv(URL)
df.head(5)


# First, let's try to see how the FlightNumber (indicating the continuous launch attempts.) and Payload variables would affect the launch outcome.
# 
# We can plot out the FlightNumber vs. PayloadMassand overlay the outcome of the launch. We see that as the flight number increases, the first stage is more likely to land successfully. The payload mass is also important; it seems the more massive the payload, the less likely the first stage will return.

# In[80]:


sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()


# We see that different launch sites have different success rates. CCAFS LC-40, has a success rate of 60 %, while KSC LC-39A and VAFB SLC 4E has a success rate of 77%.

# Next, let's drill down to each site visualize its detailed launch records.

# In[81]:


### TASK 1: Visualize the relationship between Flight Number and Launch Site
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect=2.5)
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Launch Site)", fontsize=20)
plt.show()


# Use the function catplot to plot FlightNumber vs LaunchSite, set the parameter x parameter to FlightNumber,set the y to Launch Site and set the parameter hue to 'class'

# In[82]:


# Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value
sns.scatterplot(data=df, x="FlightNumber", y="LaunchSite", hue="Class")
plt.xlabel("Flight Number", fontsize=12)
plt.ylabel("Launch Site", fontsize=12)
plt.title("Flight Number vs Launch Site", fontsize=14)
plt.show()


# Now try to explain the patterns you found in the Flight Number vs. Launch Site scatter point plots.

# In[83]:


### TASK 2: Visualize the relationship between Payload and Launch Site
sns.catplot(y="LaunchSite", x="PayloadMass", hue="Class", data=df, aspect=2.5)
plt.xlabel("Payload Mass (kg)", fontsize=20)
plt.ylabel("Launch Site", fontsize=20)
plt.show()


# We also want to observe if there is any relationship between launch sites and their payload mass.

# In[84]:


# Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value

sns.scatterplot(data=df, x="PayloadMass", y="LaunchSite", hue="Class")
plt.xlabel("Payload Mass (kg)", fontsize=12)
plt.ylabel("Launch Site", fontsize=12)
plt.title("Payload Mass vs Launch Site", fontsize=14)
plt.show()


# Now if you observe Payload Vs. Launch Site scatter point chart you will find for the VAFB-SLC launchsite there are no rockets launched for heavypayload mass(greater than 10000).
# 

# In[85]:


### TASK  3: Visualize the relationship between success rate of each orbit type

orbit_success_rate = df.groupby('Orbit')['Class'].mean().reset_index()

sns.barplot(data=orbit_success_rate, x='Orbit', y='Class')
plt.xlabel('Orbit Type', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.title('Success Rate of Each Orbit Type', fontsize=14)
plt.show()


# Next, we want to visually check if there are any relationship between success rate and orbit type.

# In[86]:


#Let's create a bar chart for the sucess rate of each orbit


# In[87]:


orbit_success_rate = df.groupby('Orbit')['Class'].mean().reset_index()

sns.barplot(data=orbit_success_rate, x='Orbit', y='Class')
plt.xlabel('Orbit Type', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.title('Success Rate of Each Orbit Type', fontsize=14)
plt.show()


# In[88]:


# HINT use groupby method on Orbit column and get the mean of Class column


# Analyze the ploted bar chart try to find which orbits have high sucess rate.

# In[89]:


### TASK  4: Visualize the relationship between FlightNumber and Orbit type

sns.catplot(data=df, x='FlightNumber', y='Orbit', aspect=2.5)
plt.xlabel('Flight Number', fontsize=12)
plt.ylabel('Orbit Type', fontsize=12)
plt.title('Flight Number vs Orbit Type', fontsize=14)
plt.show()


# In[90]:


#For each orbit, we want to see if there is any relationship between FlightNumber and Orbit type.


# Group the data by Orbit type
grouped_data = df.groupby('Orbit')

# Iterate over each orbit type
for orbit, data in grouped_data:
    # Create a scatter plot for the current orbit type
    sns.scatterplot(data=data, x='FlightNumber', y='Orbit')
    plt.xlabel('Flight Number', fontsize=12)
    plt.ylabel('Orbit Type', fontsize=12)
    plt.title(f'Flight Number vs Orbit Type ({orbit})', fontsize=14)
    plt.show()


# In[91]:


# Plot a scatter point chart with x axis to be FlightNumber and y axis to be the Orbit, and hue to be the class value
sns.scatterplot(data=df, x="FlightNumber", y="Orbit", hue="Class")
plt.xlabel("Flight Number", fontsize=12)
plt.ylabel("Orbit", fontsize=12)
plt.title("Flight Number vs Orbit", fontsize=14)
plt.show()


# We see that in the LEO orbit the Success appears related to the number of flights; on the other hand, there seems to be no relationship between flight number when in GTO orbit.

# In[92]:


### TASK  5: Visualize the relationship between Payload and Orbit type
sns.catplot(y="Orbit", x="PayloadMass", hue="Class", data=df, aspect=2)
plt.xlabel("Payload Mass (kg)", fontsize=20)
plt.ylabel("Orbit Type", fontsize=20)
plt.show()


# 
# Similarly, we can plot the Payload vs. Orbit scatter point charts to reveal the relationship between Payload and Orbit type
# 

# In[93]:


# Plot a scatter point chart with x axis to be Payload and y axis to be the Orbit, and hue to be the class value
sns.scatterplot(data=df, x="PayloadMass", y="Orbit", hue="Class")
plt.xlabel("Payload Mass (kg)", fontsize=12)
plt.ylabel("Orbit", fontsize=12)
plt.title("Payload Mass vs Orbit", fontsize=14)
plt.show()


# With heavy payloads the successful landing or positive landing rate are more for Polar,LEO and ISS.
# 
# However, for GTO we cannot distinguish this well as both positive landing rate and negative landing(unsuccessful mission) are here.

# In[94]:


### TASK  6: Visualize the launch success yearly trend
# add year column
df["Year"] = pd.DatetimeIndex(df["Date"]).year.astype(int)

df_year = df.groupby(df['Year'], as_index=False).agg({"Class": "mean"})
#df_orbit
sns.lineplot(y="Class", x="Year", data=df_year)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Success Rate", fontsize=20)
plt.show()


# We can plot a line chart with x axis to be Year and y axis to be average success rate, to get the average launch success trend.
# 

# The function will help us get the year from the date:

# In[95]:


# Plot a line chart with x axis to be the extracted year and y axis to be the success rate


# In[96]:


# A function to Extract years from the date 
year=[]
def Extract_year():
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year
Extract_year()
df['Date'] = year
df.head()
    


# we can observe that the sucess rate since 2013 kept increasing till 2020

# In[97]:


## Features Engineering


# By now,I obtained some preliminary insights about how each important variable would affect the success rate, we will select the features that will be used in success prediction in the future module.

# In[98]:


features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial','Class']]
features.head()


# In[99]:


### TASK  7: Create dummy variables to categorical columns
features_one_hot = pd.get_dummies(features[['Orbit', 'LaunchSite', 'LandingPad', 'Serial']])
features_one_hot = pd.concat([features[['FlightNumber', 'PayloadMass', 'Flights','GridFins', 'Reused', 'Legs', 'Block', 'ReusedCount', 'Class']], features_one_hot], axis=1)
features_one_hot.head(10)


# Use the function get_dummies and features dataframe to apply OneHotEncoder to the column Orbits, LaunchSite, LandingPad, and Serial. Assign the value to the variable features_one_hot, display the results using the method head. My result dataframe must include all features including the encoded ones.

# In[100]:


### TASK  8: Cast all numeric columns to `float64`
features_one_hot = features_one_hot.astype(float)
features_one_hot.dtypes


# In[101]:


features_one_hot.to_csv('dataset_part_3.csv', index=False)


# # Launch Sites Locations Analysis with Folium

# In[102]:


get_ipython().system('pip3 install folium')
get_ipython().system('pip3 install wget')
import folium
import wget
get_ipython().system('pip install folium==0.8.3')


# In[103]:


pip install folium --upgrade


# In[104]:


# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon


# # Task 1: Mark all launch sites on a map

# First, let's try to add each site's location on a map using site's latitude and longitude coordinates

# The following dataset with the name spacex_launch_geo.csv is an augmented dataset with latitude and longitude added for each site.

# In[105]:


# Download and read the `spacex_launch_geo.csv`
spacex_csv_file = wget.download('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv')
spacex_df=pd.read_csv(spacex_csv_file)


# Now, we can take a look at what are the coordinates for each site.

# In[106]:


# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df


# Above coordinates are just plain numbers that can not give you any intuitive insights about where are those launch sites. If you are very good at geography, you can interpret those numbers directly in your mind. If not, that's fine too. Let's visualize those locations by pinning them on a map.
# 
# We first need to create a folium Map object, with an initial center location to be NASA Johnson Space Center at Houston, Texas.

# In[107]:


# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)


# We could use folium.Circle to add a highlighted circle area with a text label on a specific coordinate. For example,

# In[108]:


# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)


# and you should find a small yellow circle near the city of Houston and you can zoom-in to see a larger circle.
# 

# 
# Now, let's add a circle for each launch site in data frame launch_sites
# 
# TODO: Create and add folium.Circle and folium.Marker for each launch site on the site map
# 
# An example of folium.Circle:
# 
# folium.Circle(coordinate, radius=1000, color='#000000', fill=True).add_child(folium.Popup(...))
# 
# An example of folium.Marker:
# 
# folium.map.Marker(coordinate, icon=DivIcon(icon_size=(20,20),icon_anchor=(0,0), html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'label', ))

# The generated map with marked launch sites should look similar to the following:

# In[109]:


# Initial the map
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)
# For each launch site, add a Circle object based on its coordinate (Lat, Long) values. In addition, add Launch site name as a popup label
# Initialize the map
site_map = folium.Map(location=nasa_coordinate, zoom_start=4)
# For each launch site, add a Circle object based on its coordinate (Lat, Long) values. In addition, add Launch site name as a popup label
for index, row in launch_sites_df.iterrows():
    coordinate = [row['Lat'], row['Long']]
    folium.Circle(coordinate, radius=1000, color='#000000', fill=True).add_child(folium.Popup(row['Launch Site'])).add_to(site_map)
    folium.map.Marker(coordinate, icon=DivIcon(icon_size=(20,20),icon_anchor=(0,0), html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % row['Launch Site'], )).add_to(site_map)
site_map


# Now, you can explore the map by zoom-in/out the marked areas , and try to answer the following questions:
# 
# 1. Are all launch sites in proximity to the Equator line?
# 2. Are all launch sites in very close proximity to the coast?
# Also please try to explain your findings.

# 1. no, because based on the provided code and map visualizations, it appears that the launch sites are not all located directly on the Equator, but they are generally in proximity to it.
# 
# 
# 2. yes, because the launch sites vary in their proximity to the coast. Some are close to the coast, while others are not. This variation is due to different considerations, such as safety and operational requirements.

# # Task 2: Mark the success/failed launches for each site on the map

# Next, let's try to enhance the map by adding the launch outcomes for each site, and see which sites have high success rates. Recall that data frame spacex_df has detailed launch records, and the class column indicates if this launch was successful or not

# In[110]:


spacex_df.tail(10)


# Next, let's create markers for all launch records. If a launch was successful (class=1), then we use a green marker and if a launch was failed, we use a red marker (class=0)
# 
# Note that a launch only happens in one of the four launch sites, which means many launch records will have the exact same coordinate. Marker clusters can be a good way to simplify a map containing many markers having the same coordinate.
# 
# Let's first create a MarkerCluster object

# In[111]:


marker_cluster = MarkerCluster()


# TODO: Create a new column in launch_sites dataframe called marker_color to store the marker colors based on the class value

# In[112]:


# Function to assign color to launch outcome
def assign_marker_color(launch_outcome):
    if launch_outcome == 1:
        return 'green'
    else:
        return 'red'
    
spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)
spacex_df.tail(10)


# TODO: For each launch result in spacex_df data frame, add a folium.Marker to marker_cluster

# In[113]:


# Add marker_cluster to current site_map
site_map.add_child(marker_cluster)

# for each row in spacex_df data frame
# create a Marker object with its coordinate
# and customize the Marker's icon property to indicate if this launch was successed or failed, 
# e.g., icon=folium.Icon(color='white', icon_color=row['marker_color']
for index, row in spacex_df.iterrows():
    # create and add a Marker cluster to the site map
    coordinate = [row['Lat'], row['Long']]
    folium.map.Marker(coordinate, icon=folium.Icon(color='white',icon_color=row['marker_color'])).add_to(marker_cluster)
site_map


# TASK 3: Calculate the distances between a launch site to its proximities

# Next, we need to explore and analyze the proximities of launch sites.
# 
# Let's first add a MousePosition on the map to get coordinate for a mouse over a point on the map. As such, while you are exploring the map, you can easily find the coordinates of any points of interests (such as railway)

# In[114]:


# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)
site_map


# Now zoom in to a launch site and explore its proximity to see if you can easily find any railway, highway, coastline, etc. Move your mouse to these points and mark down their coordinates (shown on the top-left) in order to the distance to the launch site.
# 
# You can calculate the distance between two points on the map based on their Lat and Long values using the following method:

# In[115]:


from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


# TODO: Mark down a point on the closest coastline using MousePosition and calculate the distance between the coastline point and the launch site.

# In[116]:


# find coordinate of the closet coastline
# e.g.,: Lat: 28.56367  Lon: -80.57163
# distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)
launch_site_lat = 28.563197
launch_site_lon = -80.576820
coastline_lat = 28.56334
coastline_lon = -80.56799
distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)
print(distance_coastline,' km')


# TODO: After obtained its coordinate, create a folium.Marker to show the distance

# In[117]:


# Create and add a folium.Marker on your selected closest coastline point on the map
# Display the distance between coastline point and launch site using the icon property 
# for example
# distance_marker = folium.Marker(
#    coordinate,
#    icon=DivIcon(
#        icon_size=(20,20),
#        icon_anchor=(0,0),
#        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance),
#        )
#    )
distance_marker = folium.Marker(
   [coastline_lat, coastline_lon],
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coastline),
       )
   )
site_map.add_child(distance_marker)


# TODO: Draw a PolyLine between a launch site to the selected coastline point

# In[118]:


# Create a `folium.PolyLine` object using the coastline coordinates and launch site coordinate
# lines=folium.PolyLine(locations=coordinates, weight=1)
coordinates = [[launch_site_lat,launch_site_lon],[coastline_lat,coastline_lon]]
lines=folium.PolyLine(locations=coordinates, weight=1)
site_map.add_child(lines)


# In[119]:


# Create a marker with distance to a closest city, railway, highway relative to CCAFS SLC-40
# Draw a line between the marker to the launch site
closest_highway = 28.56335, -80.57085
closest_railroad = 28.57206, -80.58525
closest_city = 28.10473, -80.64531


# In[120]:


distance_highway = calculate_distance(launch_site_lat, launch_site_lon, closest_highway[0], closest_highway[1])
print('distance_highway =',distance_highway, ' km')
distance_railroad = calculate_distance(launch_site_lat, launch_site_lon, closest_railroad[0], closest_railroad[1])
print('distance_railroad =',distance_railroad, ' km')
distance_city = calculate_distance(launch_site_lat, launch_site_lon, closest_city[0], closest_city[1])
print('distance_city =',distance_city, ' km')


# In[121]:


# closest highway marker
distance_marker = folium.Marker(
   closest_highway,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_highway),
       )
   )
site_map.add_child(distance_marker)
# closest highway line
coordinates = [[launch_site_lat,launch_site_lon],closest_highway]
lines=folium.PolyLine(locations=coordinates, weight=1)
site_map.add_child(lines)

# closest railroad marker
distance_marker = folium.Marker(
   closest_railroad,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_railroad),
       )
   )
site_map.add_child(distance_marker)
coordinates = [[launch_site_lat,launch_site_lon],closest_railroad]
lines=folium.PolyLine(locations=coordinates, weight=1)
site_map.add_child(lines)

# closest city marker
distance_marker = folium.Marker(
   closest_city,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_city),
       )
   )
site_map.add_child(distance_marker)
# closest city line
coordinates = [[launch_site_lat,launch_site_lon],closest_city]
lines=folium.PolyLine(locations=coordinates, weight=1)
site_map.add_child(lines)


# Launch Sites and Railways:
# To determine if launch sites are in close proximity to railways, you would need to measure the distance between each launch site and nearby railway lines. If the plotted distance lines indicate that launch sites are generally located close to railways, it suggests a potential logistical advantage for transporting materials and equipment needed for launches. This could lead to cost savings and efficient transportation.
# 
# Launch Sites and Highways:
# Similarly, analyze the distance between launch sites and nearby highways. If launch sites are situated near highways according to the plotted lines, it might signify convenient access for transportation of personnel, payloads, and equipment. This accessibility could also lead to reduced travel times and costs.
# 
# Launch Sites and Coastline:
# For determining the proximity of launch sites to coastlines, assess the plotted distances between launch sites and the nearest coastlines. If you find that launch sites are close to coastlines, it could indicate a strategic decision to facilitate overwater launches. Such launches might be advantageous due to safety reasons (minimizing the risk to populated areas) or the ability to reach specific orbits more efficiently.
# 
# Launch Sites and Cities:
# To ascertain whether launch sites maintain a certain distance from cities, examine the plotted distances between launch sites and nearby urban areas. If launch sites are located farther away from cities, it could indicate a regulatory or safety requirement to minimize the potential impact on densely populated areas. This could also be a reflection of noise, safety, or airspace regulations.
# 
# In your explanations of these findings, consider the following points:
# 
# Logistical Advantages: If launch sites are close to railways and highways, mention the potential benefits in terms of cost-effective transportation and efficient movement of materials.
# Safety and Accessibility: If launch sites are near coastlines, discuss the rationale behind this choice, such as safety considerations for overwater launches and access to specific orbital trajectories.
# Regulations and Environmental Impact: If launch sites are situated away from cities, discuss the regulatory reasons behind this decision, including noise and safety concerns, as well as the desire to minimize the impact on urban environments.

# # My findings

# As previously mentioned, launch sites are situated near the equator to capitalize on the Earth's rapid eastward rotation, which moves at approximately 30 km/s. This rotation aids spacecraft in reaching orbit more efficiently, leading to reduced fuel consumption.
# 
# Launch sites are also strategically positioned close to coastlines to enable launches over open ocean waters. This approach is guided by two primary safety considerations. Firstly, in the event of a launch abort, having the option to perform a water landing provides an added layer of crew safety. Secondly, launching over the ocean minimizes the potential risk to both individuals and property on the ground from any debris that might fall during launch or in early flight phases.
# 
# In addition, the proximity of launch sites to highways is carefully planned, facilitating the smooth transportation of essential personnel and equipment required for launches. This proximity ensures swift access and minimizes logistical challenges.
# 
# Similarly, launch sites are situated in close proximity to railways, offering the capability to transport heavy cargo and equipment needed for launches. This arrangement streamlines the movement of materials, contributing to efficient launch operations.
# 
# Contrastingly, launch sites are intentionally located at a distance from populated urban areas. This decision is guided by the goal of reducing potential risks to densely populated regions. By maintaining a safe distance from cities, launch operations minimize potential dangers to local communities and their inhabitants.

# In[122]:


get_ipython().system('pip install dash')
get_ipython().system('pip install dash==1.19.0')
get_ipython().system('pip install jupyter_dash')
get_ipython().system('pip install --upgrade plotly')


# In[123]:


from jupyter_dash.comms import _send_jupyter_config_comm_request
_send_jupyter_config_comm_request()


# In[124]:


get_ipython().system('pip list')


# In[125]:


import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output
import plotly.graph_objects as go


# In[127]:


# bug
JupyterDash.infer_jupyter_proxy_config()


# In[ ]:


# run again
JupyterDash.infer_jupyter_proxy_config()


# In[ ]:


spacex_df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv')


# In[ ]:


# Read the airline data into pandas dataframe
spacex_df = pd.read_csv('C:\\Users\\FARIA\\Downloads\\archive (5)\\spacex_launch_dash.csv')
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    
     # TASK 1: Add a dropdown list to enable Launch Site selection
    # The default select value is for ALL sites
    # dcc.Dropdown(id='site-dropdown',...)
    dcc.Dropdown(
        id='site-dropdown',
        options=[
            {'label': 'All Sites', 'value': 'ALL'},
            {'label': 'Cape Canaveral Launch Complex 40 (CAFS LC-40)', 'value': 'CCAFS LC-40'},
            {'label': 'Cape Canaveral Space Launch Complex 40 (CCAFS SLC-40)', 'value': 'CCAFS SLC-40'},
            {'label': 'Kennedy Space Center Launch Complex 39A (KSC LC-39A)', 'value': 'KSC LC-39A'},
            {'label': 'Vandenberg Air Force Base Space Launch Complex (VAFB SLC-4E)', 'value': 'VAFB SLC-4E'}
        ],
        value='ALL',
        placeholder="Select a Launch Site",
        searchable=True
    ),

    html.Br(),

    # TASK 2: Add a pie chart to show the total successful launches count for all sites
    # If a specific launch site was selected, show the Success vs. Failed counts for the site
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),

    html.P("Payload range (Kg):"),
    # TASK 3: Add a slider to select payload range
    # dcc.RangeSlider(id='payload-slider',...)

    dcc.RangeSlider(id='payload-slider',
                    min=0, max=10000, step=1000,
                    # marks={0: '0', 100: '100'},
                    value=[min_payload, max_payload]),

    # TASK 4: Add a scatter chart to show the correlation between payload and launch success
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
    ])


# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
# Place to add @app.callback Decorator
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(entered_site):
    filtered_df = spacex_df.groupby(['Launch Site'], as_index=False).mean()
    if entered_site == 'ALL':
        return px.pie(filtered_df, values='class', names='Launch Site', title='Launch Success Rate For All Sites')
    # return the outcomes in pie chart for a selected site
    filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
    filtered_df['outcome'] = filtered_df['class'].apply(lambda x: 'Success' if (x == 1) else 'Failure')
    filtered_df['counts'] = 1
    return px.pie(filtered_df, values='counts', names='outcome', title='Launch Success Rate For ' + entered_site)


# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              [Input(component_id='site-dropdown', component_property='value'),
               Input(component_id="payload-slider", component_property="value")])
def get_scatter_chart(entered_site, slider):
    filtered_df = spacex_df[
        (slider[0] <= spacex_df['Payload Mass (kg)']) & (spacex_df['Payload Mass (kg)'] <= slider[1])
    ]
    if entered_site == 'ALL':
        return px.scatter(filtered_df,
                          x='Payload Mass (kg)', y='class',
                          color='Booster Version Category',
                          title='Launch Success Rate For All Sites')
    # return the outcomes in pie chart for a selected site
    filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
    filtered_df['outcome'] = filtered_df['class'].apply(lambda x: 'Success' if (x == 1) else 'Failure')
    filtered_df['counts'] = 1
    return px.scatter (filtered_df,
                       x='Payload Mass (kg)', y='class',
                       color='Booster Version Category',
                       title='Launch Success Rate For ' + entered_site)


# Run the app
if __name__ == '__main__':
    app.run_server()

# Finding Insights Visually
# Now with the dashboard completed, you should be able to use it to analyze SpaceX launch data, and answer the following questions:
#
# Which site has the largest successful launches? KSC LC-39A
# Which site has the highest launch success rate? KSC LC-39A (success rate 76.9%)
# Which payload range(s) has the highest launch success rate? 2000-4000
# Which payload range(s) has the lowest launch success rate? 6000-8000
# Which F9 Booster version (v1.0, v1.1, FT, B4, B5, etc.) has the highest
# launch success rate? B5 (only one successful start), apart from that FT (15 successes, 8 failures)


# Based on the insights derived from the dashboard analysis of SpaceX launch data:
# 
# The launch site with the largest number of successful launches is KSC LC-39A.
# 
# The launch site with the highest launch success rate is also KSC LC-39A, with a success rate of 76.9%.
# 
# The payload range that exhibits the highest launch success rate is the range of 2000-4000 units (units need to be specified, e.g., kilograms).
# 
# The payload range with the lowest launch success rate is the range of 6000-8000 units.
# 
# Among the F9 Booster versions, the one with the highest launch success rate is "B5," with a single successful launch. Excluding that, the "FT" version holds the highest success rate, with 15 successful launches and 8 failures.

# # Assignment: Machine Learning Prediction

# In[9]:


def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 


# In[10]:


import requests
import pandas as pd


# In[11]:


data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')
data.head()


# In[12]:


get_ipython().system('pip install requests')


# In[13]:


import io


# In[14]:


import requests
import pandas as pd

URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
response = requests.get(URL1)

if response.status_code == 200:
    data = pd.read_csv(io.BytesIO(response.content))
    # Now you can work with 'data' DataFrame
else:
    print("Failed to fetch data from the URL.")


# In[15]:


# from js import fetch
# import io

# URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
# resp1 = await fetch(URL1)
# text1 = io.BytesIO((await resp1.arrayBuffer()).to_py())
# data = pd.read_csv(text1)


# In[16]:


data.head(10)


# In[18]:


# If you were unable to complete the previous lab correctly you can uncomment and load this csv
X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_3.csv')

X = data.reset_index().drop(['index'], axis=1)
X.head(100)


# In[19]:


import pandas as pd

# Load data from a CSV file into df_fe
df_fe = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_3.csv')

# Reset the index and drop the 'index' column
X = df_fe.reset_index().drop(['index'], axis=1)

# Display the first 100 rows of the resulting DataFrame
X.head(100)


# In[ ]:


# URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
# resp2 = await fetch(URL2)
# text2 = io.BytesIO((await resp2.arrayBuffer()).to_py())
# X = pd.read_csv(text2)


# In[20]:


data.head(100)


# # Task 1

# Create a NumPy array from the column Class in data, by applying the method to_numpy() then assign it to the variable Y,make sure the output is a Pandas series (only one bracket df['name of column']).

# In[21]:


Y = data['Class'].to_numpy() 

type(Y)  


# In[22]:




# Assuming 'data' is your DataFrame
# Create a NumPy array from the 'Class' column
Y = data[['Class']].to_numpy()

# Verify that 'Y' is a Pandas Series
print(type(Y))  # It should be <class 'numpy.ndarray'>


# # Task 2

# Standardize the data in X then reassign it to the variable X using the transform provided below.

# In[24]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[25]:


# students get this 
transform = preprocessing.StandardScaler()
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))  


# We split the data into training and testing data using the function train_test_split. The training data is divided into validation data, a second set used for training data; then the models are trained and hyperparameters are selected using the function GridSearchCV.

# # Task 3

# Use the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to 0.2 and random_state to 2. The training data and test data should be assigned to the following labels.

# In[26]:


from sklearn.model_selection import train_test_split

# Assuming you have X and Y already defined
# X contains your features, and Y contains your target variable

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Now, X_train and Y_train contain the training data, and X_test and Y_test contain the test data


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# we can see we only have 18 test samples.

# In[28]:


y_test.shape


# # Task 4

# Create a logistic regression object then create a GridSearchCV object logreg_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.

# In[29]:


parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}


# In[30]:


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()


# In[31]:


parameters = {"C":[0.01,0.1,1], 'penalty':['l2'], 'solver':['lbfgs']}  
lr = LogisticRegression()

logreg_cv = GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X, Y)
logreg_cv.best_estimator_

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# We output the GridSearchCV object for logistic regression. We display the best parameters using the data attribute best_params_ and the accuracy on the validation data using the data attribute best_score_.

# In[32]:


print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# # Task 5

# Calculate the accuracy on the test data using the method score:

# In[33]:


print('score on train data: ', logreg_cv.score(X_train, y_train))  # R² score on train data
print('score on test data : ', logreg_cv.score(X_test, y_test))  # R² score on test data


# Lets look at the confusion matrix:

# In[34]:


yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(y_test,yhat)


# Examining the confusion matrix, we see that logistic regression can distinguish between the different classes. We see that the major problem is false positives.

# # Task 6

# Create a support vector machine object then create a GridSearchCV object svm_cv with cv - 10. Fit the object to find the best parameters from the dictionary parameters.

# In[35]:


parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()


# In[36]:


parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}  # from 10^-3 to 10^3 in 6 steps with equal quotients
svm = SVC()

svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X, Y)
svm_cv.best_estimator_

print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# In[37]:


print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# #  Task 7

# Calculate the accuracy on the test data using the method score:

# In[38]:


print('score on train data: ', svm_cv.score(X_train, y_train))  # R² score on train data
print('score on test data : ', svm_cv.score(X_test, y_test))  # R² score on test data


# In[39]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(y_test,yhat)


# # Task 8

# Create a decision tree classifier object then create a GridSearchCV object tree_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.

# In[40]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()


# In[41]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()

tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X, Y)
tree_cv.best_estimator_

print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# In[42]:


print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# # Task 9

# Calculate the accuracy of tree_cv on the test data using the method score:

# In[43]:


print('score on train data: ', tree_cv.score(X_train, y_train))  # R² score on train data
print('score on test data : ', tree_cv.score(X_test, y_test))  # R² score on test data


# In[44]:


yhat = tree_cv.predict(X_test)
plot_confusion_matrix(y_test,yhat)


# # Task 10

# Create a k nearest neighbors object then create a GridSearchCV object knn_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.

# In[45]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()


# In[46]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()

knn_cv = GridSearchCV(KNN, parameters, cv=10)
knn_cv.fit(X, Y)
knn_cv.best_estimator_

print("tuned hpyerparameters :(best parameters) ", knn_cv.best_params_)
print("accuracy :", knn_cv.best_score_)


# In[47]:


print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)


# # Task 11

# Calculate the accuracy of knn_cv on the test data using the method score:

# In[48]:


print('score on train data: ', knn_cv.score(X_train, y_train))  # R² score on train data
print('score on test data : ', knn_cv.score(X_test, y_test))  # R² score on test data


# # We can plot the confusion matrix

# In[49]:


yhat = knn_cv.predict(X_test)
plot_confusion_matrix(y_test,yhat)


# # Task 12

# Find the method performs best:

# In[50]:


yhat = knn_cv.predict(X_test)
plot_confusion_matrix(y_test,yhat)


# Scores on test data for each method
# 
# Logistic Regression: 
# SVM: 
# Decision Tree: 
# KNN: 
# 
# 
# ## Conclusion: Logistic Regression and SVM deliver the best performance on test data.
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




