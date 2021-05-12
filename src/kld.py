import util
import math

#returns a KLD weighted dictionary
def KLD(list):

    kld={}
    relevant_list =[] #to store only the names of the releveant docs

    for i in list:
        relevant_list.append(i[0])
    
    

    '''
    Formula : KLD w(t) = Pn(t) * log2(Pn(t)/Pm(t))
    '''
    
    print(relevant_list)

    inv = util.get_inverted_index()

    for i in inv:
        pnt = 0 # Pn(t) probability of the term in the relevant documents
        pmt = 0 # Pm(t) probability of the term in the entire collection
        if i in relevant_list:
            for j in inv[i]:
                pnt = util.get_word_count(j, relevant_list)/util.get_total_sum_list(relevant_list)
                pmt = util.get_word_count(j, util.get_total_doc_list())/util.total_words
                kld[j] = pnt * math.log(pnt/pmt,2)
    
    return kld

    