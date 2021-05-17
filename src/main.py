import util
import tf_idf
import kld
import boosting
import time
import bo1
import pandas as pd
import threading

def kldbqt():
    #KLD term expansion
    print('calculating KLD list...')
    kld_list = kld.KLD(relevant_docs)
    # print(kld_list)
    print("kld boosting query terms...")
    expansion_list_kld = boosting.boosting_query_terms(kld_list,top_relevant_docs)
    query_mod = query
    for i in expansion_list_kld[0:10]:
        query_mod += ' '+i[0]
    print("modified query : ",query_mod)
    result_set_final = tf_idf.query(query_mod)
    print(result_set_final)

    df = pd.DataFrame(result_set_final,columns=["filename","similairy"])
    print(df)

def bo1bqt():
    print("caluculating bo1 list...")
    bo1_list = bo1.BO1(relevant_docs)
    print('bo1 boosting query terms...')
    expansion_list_bo1 = boosting.boosting_query_terms(bo1_list,top_relevant_docs)
    # print(bo1_list)


    query_mod = query
    for i in expansion_list_bo1[0:10]:
        query_mod += ' '+i[0]
    print("modified query : ",query_mod)
    result_set_final = tf_idf.query(query_mod)
    print(result_set_final)

    df = pd.DataFrame(result_set_final,columns=["filename","similairy"])
    print(df)



#get the list of documents in order of similarity
query = input('enter the query\n')
start = time.time()
result_set = tf_idf.query(query)

# query_vector = util.make_query_vector(query)

#take only the top five relevant docs
relevant_docs = result_set[0:5]
# print(relevant_docs)


df = pd.DataFrame(relevant_docs,columns=["filename","similairy"])
print(df)

#top relevant docs for boosting calculation
top_relevant_docs = result_set[0:3]



# kldbqt()
#KLD term expansion
print('calculating KLD list...')
kld_list = kld.KLD(relevant_docs)



#bo1 term expansion

# bo1bqt()



#boosting query terms
# kld_list = {"asd":1234}
# print('kld boosting query terms...')

expansion_list_kld = boosting.boosting_query_terms(kld_list,top_relevant_docs,relevant_docs)



query_mod = query
for i in expansion_list_kld[0:10]:
    query_mod += ' '+i[0]
print("modified query : ",query_mod)
result_set_final = tf_idf.query(query_mod)
print(result_set_final)

kld_time=time.time()

df = pd.DataFrame(result_set_final,columns=["filename","similairy"])
print(df)

print("caluculating bo1 list...")
bo1_list = bo1.BO1(relevant_docs)


print('bo1 boosting query terms...')
expansion_list_bo1 = boosting.boosting_query_terms(bo1_list,top_relevant_docs,relevant_docs)
# print(bo1_list)


query_mod = query
for i in expansion_list_bo1[0:10]:
    query_mod += ' '+i[0]
print("modified query : ",query_mod)
result_set_final = tf_idf.query(query_mod)
print(result_set_final)

bo1_time = time.time()

df = pd.DataFrame(result_set_final,columns=["filename","similairy"])
print(df)



# '''doing it using threading'''
# t1 = threading.Thread(target=kldbqt)
# t2 = threading.Thread(target=bo1bqt)

# t1.setDaemon(True)
# t2.setDaemon(True)

# t1.start()
# t2.start()

# t1.join()
# kld_time=time.time()

# t2.join()
# bo1_time = time.time()

# kldbqt()
# bo1bqt()


end = time.time()

print("kld time:", kld_time-start)
print("bo1 time:", bo1_time-kld_time)
print("time elapsed: ",end-start)

