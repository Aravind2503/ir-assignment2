import util

def BORDA(*args): 
    borda_terms = {}

    for l in args:
        pos = 0 
        for i in l:
            if i[0] in borda_terms:
                borda_terms[i[0]] += len(l)-pos
            else:
                borda_terms[i[0]] = len(l)-pos

            pos += 1
            

    
    return sorted(borda_terms.items(),key=lambda x:x[1],reverse=True)[0:20]
    




if __name__ == "__main__":
    BORDA([('a',1)],[('b',2)],[('c',66)])


