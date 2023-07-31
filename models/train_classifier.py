import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import pickle

def load_data(database_filepath='data/DisasterResponse.db'):
    '''
    Load data from database and split into training set and testing set

    Args:
        database_filepath: path to the database

    Returns:
        X: training set
        y: testing set
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable', con=engine)
    X = df['message']
    y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    '''
    Lemmatize, normalize case, and remove leading/trailing white space of the given text

    Args:
        text: input test

    Returns: clean tokens that were lemmed
    '''
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds ML model with pipeline

    Args: None
    
    Returns: optimized model
    '''
    # Build pipeline
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                        ('tfidf',TfidfTransformer()),
                        ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators = 5)))
                        ])
    # Get parameters to perform grid search wth cross validation
    parameters = {
        # 'clf__estimator__max_depth':[4,7],
        'clf__estimator__n_estimators':[5,7]
    }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluate the optimized model by the accuracy, precision, and recall
    '''
    y_pred = model.predict(X_test)
    for i in range(y_test.shape[1]):
        print(y_test.columns[i], ':')
        report = classification_report(y_test.iloc[:,i], y_pred[:,i])
        print(report)


def save_model(model, model_filepath='tuned_model.pkl'):
    '''Save model as pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))


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