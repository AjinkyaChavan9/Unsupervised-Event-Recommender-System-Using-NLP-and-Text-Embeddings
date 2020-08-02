# -*- coding: utf-8 -*-
"""
code.py - added a fix to UnicodeDecodeError while reading input.csv


"""

'''
Dependencies: pandas, numpy, nltk, wordtodigits, tensorflow, tensorflow_hub, inflect, scikit-learn
Installing can be done by running the following command in command prompt/terminal 
pip install -r requirements.txt
    
'''

#Importing libraries 
import pandas as pd
import numpy as np
import nltk #module for text preprocessing
import wordtodigits
import inflect #module to convert plural text into singular
import tensorflow_hub as hub #module to get pretrained tensorflow model
from sklearn.metrics.pairwise import cosine_similarity

"""# Employee Preference Data Analysis"""

Employee_Preference_Data = pd.read_csv('CCMLEmployeeData.csv')
Employee_Preference_Data

Domain = sorted(Employee_Preference_Data.Domain.unique())
#print(str(len(Domain)) + ' Total Domains')
Domain

Event1 = Employee_Preference_Data.Event1.unique()
Event1

Event2 = Employee_Preference_Data.Event2.unique()
Event2

Events = set(Event1) | set(Event2)
Events = sorted(list(Events))
#print(str(len(Events))+ ' Total Events')
Events

pd.options.display.max_colwidth=10000

"""# Preprocessing Input Events"""

input = pd.read_csv('input.csv', encoding = 'unicode_escape')
input

preprocess = pd.DataFrame(columns=['input'])
preprocess['input'] = input.iloc[:,0] #df.iloc[rows,columns]  (: means seecting all values of that dimension)
#Lower Casing
preprocess['input'] = preprocess['input'].str.lower()
preprocess

#Convert word numbers to digit
preprocess['input'] = preprocess['input'].apply(lambda row: wordtodigits.convert(row))
preprocess

#removing stopwords
nltk.download('stopwords') 
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
newStopWords = ["day","hour","month","days","months","hours"] #adding more stopwords
stop_words.extend(newStopWords)
preprocess['input'] = preprocess['input'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
preprocess

#Removing punctuation marks with python RegEx(Regular Expression) 
# [] - A set of characters
# \w - Returns a match where the string contains any word characters (characters from a to Z, digits from 0-9, and the underscore _ character)
# \s - Returns a match where the string contains a white space character
preprocess['input'] = preprocess['input'].str.replace('[^\w\s]','')  
preprocess

#removing numbers
preprocess['input'] = preprocess['input'].str.replace(r'\d+','')
preprocess

nltk.download('punkt') #module required for ngrams tokenizer

#FUNCTION to generate n-grams from sentences.
def extract_ngrams(data, num):
    n_grams = nltk.ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

preprocessUnigram = pd.DataFrame(columns=['input'])
preprocessUnigram['input'] = preprocess['input'].apply(lambda row: extract_ngrams(row, 1))
preprocessUnigram

preprocessBigram = pd.DataFrame(columns=['input'])
preprocessBigram['input'] = preprocess['input'].apply(lambda row: extract_ngrams(row, 2))
preprocessBigram

"""# Embedding of Preprocessed Input Events"""

#Loading Trained Text Embedding Model from Tensorflow
embed = hub.load("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2")

#FUNCTION to convert strings in list to lowercase
def list_to_lowercase(lst):
    lst = list(map(lambda x: x.lower(), lst))
    return lst

#smaller case domains
domain = list_to_lowercase(Domain)
domain

p = inflect.engine()

#smaller case events and then convert to singular
event = list_to_lowercase(Events)
for i in range(len(event)):
    event[i] = p.singular_noun(event[i]) 
event

#FUNCTION to Convert List to Numpy array with only 1 value in 1 row for Embedding input
def convert_LISTtoNUMPY_for_Embedding(list):
    array = np.array(list).reshape(-1,1) #shape(-1,1) -> (respective_no_of_rows, 1 column)
    return array

#Convert Domain list to Numpy array with each value in one row for Embedding input
domain_array = convert_LISTtoNUMPY_for_Embedding(domain)
domain_array

#Convert Event list to Numpy array with each value in one row for Embedding input
event_array = convert_LISTtoNUMPY_for_Embedding(event)
event_array

#FUNCTION to generate Embeddings for domain, events
def Lst_embed(Lst , array):
    Lst_embedding={}
    for i in range(len(array)):
        Lst_embedding[Lst[i]] = embed(array[i]).numpy()
    return Lst_embedding

np.set_printoptions(edgeitems=5, linewidth=100000) #adjusting np array printing option to display one row vectors in one line

# Domain names(lowercase) to vector
Domain_embedding = Lst_embed(Domain,domain_array)
Domain_embedding

Event_embedding = Lst_embed(Events, event_array)
Event_embedding

"""# Extracting Domains & Events from input"""

#FUNCTION for Domain_Recommendation for respective ngrams
def Domain_Recommendation_ngram(preprocess_ngram, threshold):
    Domain_Recommendation_ngram = {}
    for j in range(len(preprocess_ngram['input'])):
        word2vec_input_event = embed(preprocess_ngram['input'][j]).numpy()
        domain_recommend = [] #list to keep domains matched to input event
        for i in range(len(Domain)):
            similarity_index = cosine_similarity(word2vec_input_event, Domain_embedding[Domain[i]])
            for value in np.nditer(similarity_index):
                if value >= threshold :
                    domain_recommend.append(Domain[i])
                    break
        if not domain_recommend:
            domain_recommend.append('Other') #if no domain recommended
        Domain_Recommendation_ngram[input.iloc[:,0][j]] = domain_recommend
    return Domain_Recommendation_ngram

Domain_Recommendation_Bigram = Domain_Recommendation_ngram(preprocessBigram, 0.5)
Domain_Recommendation_Bigram

Domain_Recommendation_Unigram = Domain_Recommendation_ngram(preprocessUnigram,0.5)
Domain_Recommendation_Unigram

def Combine_Unigram_Bigram(unigram,bigram):
    ds = [unigram, bigram]
    d = {}
    for k in unigram.keys():
        d[k] = np.unique(np.concatenate(list(d[k] for d in ds))).tolist()
    return d

Domain_Recommendation = Combine_Unigram_Bigram(Domain_Recommendation_Unigram, Domain_Recommendation_Bigram)
Domain_Recommendation

#FUNCTION for Event_Recommendation for respective ngrams
def Event_Recommendation_ngram(preprocess_ngram, threshold):
    Event_Recommendation_ngram = {}
    for j in range(len(preprocess_ngram['input'])):
        word2vec_input_event = embed(preprocess_ngram['input'][j]).numpy()
        event_recommend = [] #list to keep event_type matched to input text event
        for i in range(len(Events)):
            similarity_index = cosine_similarity(word2vec_input_event, Event_embedding[Events[i]])
            for value in np.nditer(similarity_index):
                if value >= threshold :
                    event_recommend.append(Events[i])
                    break
        Event_Recommendation_ngram[input.iloc[:,0][j]] = event_recommend
    return Event_Recommendation_ngram

Event_Recommendation_Bigram = Event_Recommendation_ngram(preprocessBigram, 0.5)
Event_Recommendation_Bigram

Event_Recommendation_Unigram = Event_Recommendation_ngram(preprocessUnigram,0.5)
Event_Recommendation_Unigram

Event_Recommendation = Combine_Unigram_Bigram(Event_Recommendation_Unigram, Event_Recommendation_Bigram)
Event_Recommendation

"""# Mapping Recommendations with Employees and their preference"""

Employee_Preference_Data

def find_employees_domain(recommendeddomains):
    return(Employee_Preference_Data['Name'].loc[Employee_Preference_Data['Domain'].isin(recommendeddomains)])

def find_employees_event(recommendedevents):
    return(Employee_Preference_Data['Name'].loc[Employee_Preference_Data['Event1'].isin(recommendedevents) | Employee_Preference_Data['Event2'].isin(recommendedevents)])

def pandas_to_string(employee_names_df):
    return ', '.join(employee_names_df.values.flatten().tolist())

#Mapping Recommendations with Employees and their preference
Recommendations = pd.DataFrame(columns=['input','Recommended_Employees'], index=input.index) #Empty dataframe with same no. of rows as no. of input events
for i in range(len(input)):
    Recommendations['input'][i] = input.iloc[i,0]
    recommended_domains = Domain_Recommendation[input.iloc[i,0]] # iloc[rows,columns]
    recommended_events = Event_Recommendation[input.iloc[i,0]]
    if (recommended_domains == ['Other']): #condition when no domain detected
        Recommendations['Recommended_Employees'][i] = pandas_to_string(find_employees_event(recommended_events)) #direct recommendation with event names
    else:
        DomainMatchedIndex = find_employees_domain(recommended_domains).index # Getting index of employees with recommended domains
        EventsMatchedIndex = find_employees_event(recommended_events).index #Getting index of employees with recommended domains
        RecommendationIndex = DomainMatchedIndex.intersection(EventsMatchedIndex) #Intersection to get Respective Domain Employees with their Preferred events
        Recommendations['Recommended_Employees'][i] = pandas_to_string(Employee_Preference_Data['Name'][RecommendationIndex])

Recommendations

"""### Exporting Recommendations to Spreadsheet(xls)"""

Recommendations.to_excel('output.xls', index=False)
