import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        load message and category files and returns merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, left_on='id', right_on='id')
    return df

    
def clean_data(df):
    """
        clean the data frame.
        - rename the category columns.
        - convert category value to numeric
        - drop duplicates

        returns cleaned dataframe.
    """
    categories = df.categories.str.split(";", expand=True)
    categories.head()

    row = categories.iloc[0, :]
    category_colnames = row.str.split("-", expand=True)[0]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast='integer')


    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df
    
def save_data(df, database_filename):
    """
        Save dataframe to `Message` table in sqlite file at database_filename
        if `Message` table is exists, replace it.
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("Message", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()