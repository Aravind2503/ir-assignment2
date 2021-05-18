import util
import tf_idf
import kld
import boosting
import time
import bo1
import pandas as pd
import threading
import borda


#get the list of documents in order of similarity
query = input('enter the query\n')
start = time.time()
result_set = tf_idf.query(query)

# query_vector = util.make_query_vector(query)

#take only the top five relevant docs
relevant_docs = result_set[0:5]
# print(relevant_docs)


df = pd.DataFrame(result_set,columns=["filename","similairy"])
print(df)

#top relevant docs for boosting calculation
top_relevant_docs = result_set[0:3]



# kldbqt()
#KLD term expansion
print('calculating KLD list...')
kld_list = kld.KLD(relevant_docs)


print('kld boosting query terms...')

expansion_list_kld = boosting.boosting_query_terms(kld_list,top_relevant_docs,relevant_docs)



kld_time=time.time()



print("caluculating bo1 list...")
bo1_list = bo1.BO1(relevant_docs)


print('bo1 boosting query terms...')
expansion_list_bo1 = boosting.boosting_query_terms(bo1_list,top_relevant_docs,relevant_docs)
# print(bo1_list)



bo1_time = time.time()





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

#borda voting
kld_list1 = sorted(kld_list.items(),key=lambda x:x[1],reverse=True)[0:20]
bo1_list1 = sorted(bo1_list.items(),key=lambda x:x[1],reverse=True)[0:20]
expansion_list_kld1 = expansion_list_kld[0:20]
expansion_list_bo11 =  expansion_list_bo1[0:20]

final_expansion_terms = borda.BORDA(kld_list1,bo1_list1,expansion_list_kld1,expansion_list_bo11)

borda_voting_time = time.time()

print(final_expansion_terms)



querymod = query
for i,_ in final_expansion_terms:
    querymod += " "+i

result_set_final = tf_idf.query(querymod)

df = pd.DataFrame(result_set_final,columns=["filename","similairy"])
print(df)

end = time.time()



print("kldbqt time: ", kld_time-start)
print("bo1bqt time: ", bo1_time-kld_time)
print("borda time: ",borda_voting_time-bo1_time)
print("time elapsed: ",end-start)



