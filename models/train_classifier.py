import sys
import string
import pickle

import pandas as pd

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
        load sqlite database from database_filepath
        and make dataframe from `Message` table
        returns
        - X: message text list
        - Y: corresponding category list
        - columns: labels of Y
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("Message", engine)
    X = df.message
    Y = df.iloc[:, 5:]
    return X, Y, list(Y.columns)


# skip tokens with custom list
skip_tokens = stopwords.words('english') + ["'s", "n't"] + list(string.punctuation)

def tokenize(text):
    """
        tokenize the text and returns list of token
        - tokenize
        - lemmatize
        - normalize
        - stop words filtering
        - punctuation filtering
    """

    lemm = WordNetLemmatizer()
    
    text = word_tokenize(text)
    text = [lemm.lemmatize(x).lower().strip() for x in text]
    text = filter(lambda x: x not in skip_tokens, text)
    
    return " ".join(text)
    

def build_model():
    """
        Build a tf-idf/random forest model and find hyperparameters using GridSearchCV
        returns optimized model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # params dict to tune a model
    # Some parameters are commented because it takes too much time.
    parameters = {
    #     'clf__estimator__max_depth': [10, 25, 50],
    #     'clf__estimator__n_estimators': [10, 50, 100],
    #     'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (5000, 10000, 50000)
    }

    # instantiate a gridsearchcv object with the params defined
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1) 

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        evaludate the model by X_test, Y_Test dataset
        print the classification report(f1-score, recall, precision) of each column(category)
    """
    reports = []
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = category_names
    
    for column in category_names:
        report = classification_report(Y_test[column], y_pred[column])
        reports.append(report)

        print("Classification report for {}".format(column))
        print(report)


def save_model(model, model_filepath):
    """
        Save the model as a pickle file to model_filepath
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()