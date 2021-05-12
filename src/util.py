#utility functions

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

total_words = 31354 #to avoid recalculation (already calculated once)



#process lines
def process_lines(line):
    
    line = line.lower() #converts all the characters in the string to lowercase    
    line = re.sub(r'\d+', '', line) #remove all the digits from the string    
    line = line.translate(str.maketrans('','',string.punctuation)) #removing punctuation from the string    
    line = line.strip() #removing leading and trailing whitespaces

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

def make_query_vector(query):

    c_d={}
    q_vec={}
    
    processed_query = process_lines(query)

    # print(processed_query)


    for i in processed_query:
        if i not in c_d:
            c_d[i] = 1
        else:
            c_d[i] += 1

    maxi = c_d[max(c_d, key=c_d.get)]

    # print(c_d)


    #keeping the idf weights ready
    idf = get_idf()

    #getting the tf_idf weights
    tf_idf = get_tf_idf()


    for i in c_d:
        if i in idf:
            q_vec[i] = c_d[i]/maxi * idf[i]
        else:
            q_vec[i] = 0

    return q_vec

def get_tf_idf():
    with open(r'inverted_index/tf_idf.json') as f:
        tf_idf = json.load(f)
    return tf_idf

def get_idf():
    with open(r'inverted_index/idf.json') as f:
        idf = json.load(f)
    return idf

def get_inverted_index():
    with open(r'../Dataset/inverted.json') as f:
        inv = json.load(f)
    
    return inv

#returns the total number of terms in the whole collection
def get_total_sum():
    inv = get_inverted_index()

    total = 0
    for i in inv:
        for j in inv[i]:
            total += inv[i][j]
    return total

#gets the total count of the word given and in the list of the documents passed
#eg  get_relevant_count("earth", ["T4.txt","T2.txt"])
def get_word_count(word,rel_list):
    inv = get_inverted_index()

    val = 0

    for i in inv:
        if i in rel_list:
            for j in inv[i]:
                if j == word:
                    val += inv[i][j]

    return val

#gets list of all the documents in the collection
def get_total_doc_list():
    os.chdir('../Dataset')
    dir_list = os.listdir()

    final_list = []

    for i in dir_list:
        if ".txt" in i :
            final_list.append(i)
    
    return final_list


    os.chdir('../src')

#total number of words in the list of given docs
def get_total_sum_list(list):
    inv = get_inverted_index()

    total = 0
    for i in inv:
        if i in list:
            for j in inv[i]:
                total += inv[i][j]
    return total





if __name__ =='__main__':
    print(get_total_sum_list(["T4.txt","T2.txt"]))