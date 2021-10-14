# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine


def load_data(messages_filepath, categories_filepath):
    """This function used to load dataset from the given 2 links of categories & messages"""
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merging datasets
    df = pd.merge(messages, categories, how='inner', on='id')

    return df


def clean_data(df):
    """This function is used to clean dataset after merging"""
    # split categoriescategories
    categories = df['categories'].str.split(pat=';',  # pattern
                                            expand=True  # expand or not
                                            )
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # extract a list of new column names for categories, by removing the last 2 characters.
    category_colnames = row.apply(lambda x: x[:(len(x)-2)])
    # plug into column-name
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(str(x)[-1]))
        # for relate class, with value == 1 for value=2
        categories['related'] = categories['related'].apply(lambda x: 1 if x > 0 else 0)

    # drop child_alone, since it does not have any value of 1
    categories.drop(columns=['child_alone'], inplace=True)

    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filepath):
    """This function help to save dataframe to database"""

    engine = create_engine('sqlite:///' + str(database_filepath))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()