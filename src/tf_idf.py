#functions to caluculate tf_idf weight.

import nltk
import re 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import codecs
import json
import os
import math
import pandas as pd
import util




def process_lines(line):

    

    #converts all the characters in the string to lowercase
    line = line.lower()

    #remove all the digits from the string
    line = re.sub(r'\d+', '', line)

    #removing punctuation from the string
    line = line.translate(str.maketrans('','',string.punctuation))

    #removing leading and trailing whitespaces
    line = line.strip()

    #removing stopwords

    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(line)
    line = [i for i in tokens if not i in stop_words]

    #stemming using Porter Stemmer
    
    stemmer = PorterStemmer()
    
    
    for i in range(0,len(line)):
        line[i] = stemmer.stem(line[i])
    
    
    #lemmatization
    # lemmatizer = WordNetLemmatizer()
    
    # for i in range(0,len(line)):
    #     line[i] = lemmatizer.lemmatize(line[i])
    return line



def dot_product(a,b):
    product = 0 
    for i in a:
        if i in b:
            product += a[i]*b[i]
    
    return product
    
def magnitude(a):
    mag = 0
    for i in a:
        mag += a[i]**2
    
    return math.sqrt(mag)



#returns a list of all the docs for the query
def query(query):


    # query = input('enter the query string\n')
    q_vec = util.make_query_vector(query)



    #keeping the idf weights ready
    with open(r'inverted_index/idf.json') as f:
        idf = json.load(f)

    with open(r'inverted_index/tf_idf.json') as f:
        tf_idf = json.load(f)




    # print(f'query_vector :{str(q_vec)}')
    #for storing the results
    result_set = []
    for i in tf_idf:
        if dot_product(q_vec,tf_idf[i]) != 0.0:
            val = (dot_product(q_vec,tf_idf[i]))/(magnitude(q_vec)*magnitude(tf_idf[i]))
        else:
            val = 0.0
        result_set.append((i,val))

    #sorting the result_set 
    # print(sorted(result_set,key = lambda tup:tup[1],reverse=True))

    l = (sorted(result_set,key = lambda tup:tup[1],reverse=True))

    # df = pd.DataFrame(l,columns=['File','Similarity'])
    # print(df)

    return l









