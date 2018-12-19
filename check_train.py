from check_defn import *
import numpy as np
import random

def train(word_list, query_list, ensemble, iterations,l_rate,matrix,lamb):
    #counter is used to go to next ensemble
    counter = 0
    total_data = len(word_list)
    for iteration in range(iterations):
        #To keep the cost
        cost = 0
        #To store the entries for back propagation
        back = []
        #For keeping the accuracy
        accuracy = 0
        for q in range(ensemble):
            query_no = (q+counter)%total_data
            select_len = len(query_list[query_no])
            select = random.randint(1,100)%select_len
            query = query_list[query_no][select]
            #Gives a query for forward propagation
            (cost_minimum, found_word) = forward(query, word_list,matrix,lamb)
            cost+=cost_minimum
            #Backpropagation
            back.append((query, found_word, word_list[query_no]))
            if found_word == word_list[query_no]:
                accuracy+=1
        cost = cost/ensemble
        accuracy = accuracy*100/ensemble
        print('Iteration:',iteration,'\tAccuracy:',accuracy,'%')
        (correction,cost) = backward(back, matrix,lamb)
        print('\tSecondary Cost:',cost)
        matrix = matrix - l_rate*correction/(((accuracy+1)**2)*ensemble)
    counter+=ensemble
    return matrix


#Training dataset is wikipedia.txt
fp = open('wikipedia.txt','r')
#To store the words
word_list = []
#To store the queries
query_list = []
#Reads the data in the file
data = fp.readline()
while(data!=''):
    data = data.split(':')
    word_list.append(data[0])
    query_list.append(data[1].split())
    data = fp.readline()

#Initializes a matrix of all entries one of proper length
matrix = np.ones(len(create_vector(cost('a','a'))))
#Learning Rate : learning_rate
learning_rate = 1
#Ensemble or Batch size : ensemble_size
ensemble_size = len(word_list)//16
#Iterations : iterations
iterations = 32
#Lambda : lamb_da
lamb_da = 0

print('Learning rate:',learning_rate, 'Ensemble Size:', ensemble_size, 'Iterations:',iterations,'Lambda:',lamb_da)
matrix = train(word_list,query_list, ensemble_size, iterations, learning_rate, matrix, lamb_da)
print([matrix])
