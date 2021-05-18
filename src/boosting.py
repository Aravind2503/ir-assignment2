#module for implementing boosting query terms method 


import math
import util
import tf_idf
import threading

#pass a dictionary into this method
# def boosting_query_terms(d,top_docs):
    

#     #get the document prominence (i.e. similarity values of each document)
#     n = len(top_docs)
#     print(top_docs)

#     #top_docs has doc_name in 0 pos and similarity in 1 pos
#     inv = util.get_inverted_index()
#     sim = {}
#     for i in inv:
#         sim[i] = 0 #so that we account for similarity with the document and itself
#         for j in top_docs:
#             sim[i] +=  tf_idf.dot_product(inv[i],inv[j[0]])/(tf_idf.magnitude(inv[i])*tf_idf.magnitude(inv[j[0]]))/(n-1)
    
    
#     # print(sim)

#     boosted_list = {}
#     idf = util.get_idf()
#     # tf_idff = util.get_tf_idf()
    

#     for i in d:
#         boosted_list[i] = 0
#         for j in inv:
#             if i in inv[j]:
#                 boosted_list[i] += inv[j][i] * idf[i] * sim[j]
#                 # boosted_list[i] += tf_idff[j][i] * sim[j]

#         boosted_list[i] = math.log10(1+boosted_list[i])
    
#     # print(sorted(boosted_list.items(),key=lambda v: v[1],reverse=True)[0:20])

#     return sorted(boosted_list.items(),key=lambda v: v[1],reverse=True)[0:20]

def boosting_query_terms(d,top_docs,relevant_docs):
    

    #get the document prominence (i.e. similarity values of each document)
    n = len(top_docs)
    # print(top_docs)

    #top_docs has doc_name in 0 pos and similarity in 1 pos
    
    tf_idff = util.get_tf_idf()
    sim = {}
    for i in relevant_docs:
        sim[i[0]] = 0 #so that we account for similarity with the document and itself
        for j in top_docs:
            sim[i[0]] +=  tf_idf.dot_product(tf_idff[i[0]],tf_idff[j[0]])/(tf_idf.magnitude(tf_idff[i[0]])*tf_idf.magnitude(tf_idff[j[0]]))/(n-1)
    
    
    # print(sim)

    boosted_list = {}
    idf = util.get_idf()
    # tf_idff = util.get_tf_idf()
    
    inv = util.get_inverted_index()
    for i in d:
        boosted_list[i] = 0
        for j in relevant_docs:
            if i in inv[j[0]]:
                boosted_list[i] += inv[j[0]][i] * idf[i] * sim[j[0]]
                # boosted_list[i] += tf_idff[j][i] * sim[j]

        boosted_list[i] = math.log10(1+boosted_list[i])
    
    # print(sorted(boosted_list.items(),key=lambda v: v[1],reverse=True)[0:20])

    return sorted(boosted_list.items(),key=lambda v: v[1],reverse=True)[0:20]






if __name__=="__main__":
    boosting_query_terms({"a":12},[],[])
    idf = util.get_idf()
    print(idf)

    


