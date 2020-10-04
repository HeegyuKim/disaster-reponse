# Disaster Response Pipeline Project
Udacity Data Science Nanodegree - Project 1

## Setup
- Python3 required

#### Install packages
```
pip install -r requirements.txt
```

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Structure
#### data/
- read data from csv files
- data cleaning, drop duplicates
- save cleaned data to sqlite database(data/DisasterResponse.db)
#### models
- read data from database and tokenize
- train models
- save trained models as pickle to models/classifier.pkl
#### app/
- flask app for running demo in web browser
- get the models from models/classifier.pkl and use it
