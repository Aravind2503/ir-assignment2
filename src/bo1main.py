import tf_idf
import bo1
import util


query = input('Enter the query\n')
result_set = tf_idf.query(query)
query_vector = util.make_query_vector(query)
relevant_docs = result_set[0:5]
print('Calculating bo1 list..')
print(bo1.BO1(relevant_docs))

