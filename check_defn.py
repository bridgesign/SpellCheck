import numpy as np

#Computes the cost of the query with respect to word
def cost(query,word):
    #Make query and word lowercase
    query = query.lower()
    word = word.lower()

    #length cost is based on absolute difference on lenght
    length = abs(len(query)-len(word))+1

    #letter score is based on whether a letter matches
    #and at how much distance it is from given word
    #If letter is not present, cost equal to lenght of
    #query is added.
    letter = 0
    letter_score = {}
    for i in range(len(word)):
        letter_score[word[i]] = []
    for i in range(len(word)):
        letter_score[word[i]].append(i)
    for i in range(len(query)):
        if query[i] not in letter_score:
            letter+=len(query)
        else:
            m = abs(i - letter_score[query[i]][0])
            for j in letter_score[query[i]]:
                n = abs(i-j)
                if n<m:
                    m = n
            letter+=m
    #For normalization, this cost is divided by lenght of query
    letter = letter/len(query) +1

    #Cost is given based on whether the first letter, middle
    #letter and last letter match.
    #Cost of 5 if it doesn't match and 1 if matched.
    first_position = 5
    last_position = 5
    middle_position = 5
    if query[0] == word[0]:
        first_position = 1
    if query[-1] == word[-1]:
        last_position = 1
    if query[len(query)//2] == word[len(word)//2]:
        middle_position = 1

    #Frequency cost is decided by frequency of letters in
    #word and query.
    frequency = 1
    for i in query:
        frequency+= abs(query.count(i) - word.count(i))

    #Convert the word and query in sets for letter_set cost
    #letter_set cost is given by symmetric difference of sets.
    word = set(word)
    query = set(query)
    letter_set = len(word^query)+1

    #Returning the base cost vector with addition of constant 1.
    return [1,length, letter, first_position, last_position, middle_position,frequency,letter_set]

#Creates a cost vector from the values returned by cost function
def create_vector(cost_vector):
    cost_vector = [cost_vector[i]*cost_vector[j]*cost_vector[k] for i in range(8) for j in range(8) for k in range(8) if (i<=j and j<=k)]
    return np.array(cost_vector)

#Returns the word in wordlist with the least cost
def forward(query, wordlist, matrix,lamb):
    cost_minimum = 1000000000000000000000000
    found_word = ''
    for word in wordlist:
        check = ((np.matmul(matrix, create_vector(cost(query,word))))**2 + lamb*np.sum(matrix**2))/(2*len(matrix))
        if check<cost_minimum:
            cost_minimum = check
            found_word = word
    return (cost_minimum, found_word)

#Here word is the actuual meant word
#Returns the difference vector
def backward(query_list, matrix,lamb):
    m = len(matrix)
    cost_total=np.zeros(len(matrix))
    cost_t = 0
    for query in query_list:
        found = create_vector(cost(query[0],query[1]))
        actual = create_vector(cost(query[0],query[2]))
        cost_total += np.matmul(matrix,found)*found - np.matmul(matrix,actual)*actual + lamb*matrix
        cost_t+=(np.matmul(matrix,found)- np.matmul(matrix,actual))**2+lamb*np.sum(matrix**2)
    cost_t = cost_t/(2*m)
    error = cost_total/(m)
    return (error,cost_t)
