import util
import math


def BO1(list):

    bo1={}
    relevant_list =[]
    total_tf = util.get_total_tf()

    for i in list:
        relevant_list.append(i[0])
    print(relevant_list)
    inv = util.get_inverted_index()

    for i in inv:
        pc = 0 
        if i in relevant_list:
            for j in inv[i]:
                tf = util.get_word_count(j, relevant_list)
                # pc = util.get_word_count(j, relevant_list)/len(util.get_total_doc_list())
                pc = total_tf[j]/len(util.get_total_doc_list())
                bo1[j] = tf*math.log((1+pc)/pc,2)+math.log(1+pc,2)
    
    return bo1



def BO11(list):
    bo1={}

    
    return bo1
