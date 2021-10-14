# Disaster Responses Pipeline
## Project Overview
In this project, we'll apply data engineering skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) t
o build a model for an API that classifies disaster messages.

**Data**: we're provided a data set containing real messages that were sent during disaster events. 

**Output**: 
1. create a machine learning pipeline to categorize these events so that we can send the messages to 
an appropriate disaster relief agency.

2. web app where an emergency worker can input a new message and get classification results 
in several categories. The web app will also display visualizations of the data. 
Link to the webapp: [To-be-update: Link after deployment]()

Below are a few screenshots of the web app.
![disaster-response-project_webapp1](disaster-response-project_webapp1.png)

![disaster-response-project_webapp2](disaster-response-project_webapp2.png)

## Installation
This project requires Python 3.x and others libraries included in the `requirements.txt` file.

## Instructions
In a terminal that contains this README file, run commands in the following sequence:
 + `pip3 install -r requirements.txt`
 + `etl` pipeline
 + `ml` pipeline
 + `web deployment` pipeline


# Project Components
There are three components of this project.

### 1. ETL Pipeline
In a Python script, `process_data.py`, write a data cleaning pipeline that:
+ Loads the `messages` and `categories` datasets
+ Merges the two datasets
+ Cleans the data
+ Stores it in a SQLite database

### 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:
+ Loads data from the SQLite database
+ Splits the dataset into training and test sets
+ Builds a text processing and machine learning pipeline
+ Trains and tunes a model using GridSearchCV
+ Outputs results on the test set
+ Exports the final model as a pickle file

### 3. Flask Web App
An `index.html` file is provided for flask web app deployment, extra features built on knowledge of 
flask, html, css and javascript for additional touch on the front-end visualization

+ Modify file paths for database and model as needed
+ Add data visualizations using Plotly in the web app. 
+ Wrap the webapp and deploy into Heroku server.


# Project Structure
Here's the file structure of the project:

        - app
        | - template
        | |- master.html  # main page of web app
        | |- go.html  # classification result page of web app
        |- run.py  # Flask file that runs app

        - data
        |- disaster_categories.csv  # data to process 
        |- disaster_messages.csv  # data to process
        |- process_data.py
        |- InsertDatabaseName.db   # database to save clean data to

        - models
        |- train_classifier.py
        |- classifier.pkl  # saved model 

        - ETL Pipeline Preparation.ipynb # notebook file of Project Workspace - ETL
        - ML Pipeline Preparation.ipynb # notebook file of Project Workspace - Machine Learning Pipeline.
        - README.md