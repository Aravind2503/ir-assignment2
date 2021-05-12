import tf_idf
import kld
import util


#get the list of documents in order of similarity
query = input('enter the query\n')
result_set = tf_idf.query(query)

query_vector = util.make_query_vector(query)

#take only the top five relevant docs
relevant_docs = result_set[0:5]

print('calculating KLD list...')

#KLD term expansion
print(kld.KLD(relevant_docs))


