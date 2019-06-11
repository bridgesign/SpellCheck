from check_defn import *
import numpy as np

def test(file, matrix):
    accuracy = 0
    counter = 0
    test_examples = 0
    line = file.readline()
    #word list
    word_list = []
    #query list
    query_list = []
    #Reads the data from the file
    while(line!=''):
        (word,query) = line.split(':')
        #query contains different queries
        #corresponding to same word
        query = query.split()
        word_list.append(word)
        query_list.append(query)
        counter+=1
        test_examples+=len(query)
        line = file.readline()
    for i in range(counter):
        for query in query_list[i]:
            (cost_minimum, found_word) = forward(query, word_list,matrix,0)
            if found_word == word_list[i]:
                accuracy+=1
    return accuracy*100/test_examples

#Found Matrix after training
matrix = np.load('matrix.npy')


file = open('spell-testset1.txt','r')
print('Test Set: spell-testset1.txt')
print('Accuracy:',test(file,matrix),'%')
file.close()
file = open('spell-testset2.txt','r')
print('Test Set: spell-testset2.txt')
print('Accuracy:',test(file,matrix),'%')
file.close()
file = open('aspell.txt','r')
print('Test Set: aspell.txt')
print('Accuracy:',test(file,matrix),'%')
file.close()
