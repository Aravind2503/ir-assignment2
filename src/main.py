import util
import tf_idf
import kld
import boosting
import time



#get the list of documents in order of similarity
query = input('enter the query\n')
start = time.time()
result_set = tf_idf.query(query)

# query_vector = util.make_query_vector(query)

#take only the top five relevant docs
relevant_docs = result_set[0:5]
print(relevant_docs)

#top relevant docs for boosting calculation
top_relevant_docs = result_set[0:3]

print('calculating KLD list...')

#KLD term expansion
kld_list = kld.KLD(relevant_docs)
# print(kld_list)

#boosting query terms
# kld_list = {"asd":1234}
print('boosting query terms...')
expansion_list = boosting.boosting_query_terms(kld_list,top_relevant_docs)

end = time.time()

query_mod = query
for i in expansion_list:
    query_mod += ' '+i[0]
print("modified query : ",query_mod)
result_set_final = tf_idf.query(query_mod)
print(result_set_final)


print("time elapsed: ",end-start)