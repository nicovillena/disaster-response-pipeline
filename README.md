## Udacity's Data Scientist Nanodegree Project: Disaster Response Pipeline

### Project Overview

The objective of this project is to analyze real messages that were sent during disaster events by build a model for an API that categorizes these events in order to send the messages to an appropriate disaster relief agency. The data is from Figure Eight.

The project combines an ETL Pipeline, Machine learning Pipeline and web application in order to clean the data, train and test a model, and display it visually.

### Table of Contents

1. [Libraries](#libraries)
2. [Content](#contents)
3. [Instructions](#instructions)
4. [File Descriptions](#files)

### Libraries: <a name="libraries"></a>

    scikit-learn
    nltk
    Flask
    gunicorn
    numpy
    pandas
    plotly
    sqlalchemy
    jsonschema

### Contents: <a name="contents"></a>

* ETL Pipeline

        Loads the messages and categories datasets.
        Merges the two datasets.
        Cleans the data.
        Stores it in a SQLite database.

* ML Pipeline

        Loads data from the SQLite database.
        Splits the dataset into training and test sets.
        Builds a text processing and Machine Learning pipeline.
        Trains and tunes a model using GridSearchCV.
        Outputs results on the test set.
        Exports the final model as a pickle file.

* Flask Web App

        It displays results in a Flask web app which is deployed on Heroku.

### Instructions: <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Descriptions: <a name="files"></a>

    - data/process_data.py: ETL pipeline used to prepare data for model building.
    
    - models/train_classifier.py: Machine Learning pipeline, output is a trained classifier. 
    
    - models/classifier.pkl: Trained classifer, output of the Machine Learning pipeline.
    
    - app/run.py: Flask file to run the web application.


