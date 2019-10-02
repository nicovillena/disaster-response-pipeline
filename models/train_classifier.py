import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Input:
        database_filepath: File path of sql database
    Output:
        X: Messages
        Y: Categories
        category_names: List of labels for 36 categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = list(Y.columns)

    return X, Y, category_names
    
def tokenize(text):
    '''
    Input:
        text: Messages text
    Output:
        clean_tokens: Cleaned, tokenized and lemmatized text
    '''
        
    # Find and replace urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Replace puctuation
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    words = [w for w in tokens if w not in stopwords.words('english')]
    
    # Lemmatize and normalize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    '''
    Input: 
        None
    Output: 
        Results of GridSearchCV
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__min_samples_split': [2, 4]
        }

    cv = GridSearchCV(estimator = pipeline, param_grid = parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input: 
        model: Model to be evaluated
        X_test: Test data
        Y_test: Labels for test data
        category_names: Labels for 36 categories
    Output:
        Classification report and accuracy for each category
    '''
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        # Report f1 score, precision and recall
        report = classification_report(Y_test.iloc[:, i].values, Y_pred[:, i])
        
        # Calculate accuracy
        accuracy = accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])
       
        print("\n", "Category:", category_names[i], "\n", report)
        print('Accuracy of %0s: %.2f'%(category_names[i], accuracy))
    
def save_model(model, model_filepath):
    '''
    Input: 
        model: Model to be saved
        model_filepath: File path of output file
    Output:
        Saved model in pickle file
    '''
    pickle.dump(model, open(model_filepath, "wb"))

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