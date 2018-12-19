# SpellCheck
This is an attempt to use stochastic gradient descent with predefined features to calculate the distance or cost between two word queries and to use it for finding the correct spelling for the given query.

## Datasets
The data sets were accquired from https://www.kaggle.com/bittlingmayer/spelling
### Training Set
wikipedia.txt
### Test Sets
spell-testset1.txt -- Accuracy 96%\
spell-testset2.txt -- Accuracy 95%\
aspell.txt         -- Accuracy 91%
## Predefined Features
1. length - Absolute difference between length of strings
2. letter - It is based on whether letters match and at how much distance they are
3. first_position - Whether the first letter matches or not
4. last_position  - Whther last letter matches or not
5. middle_position  - Whether middle letter matches or not
6. frequency  - Based on the difference between the frequency of letters in both words
7. letter_set - Symmetric difference in the sets made from the letters of the two words

## Requirements
1. Python 3.0
2. Numpy
## Observations
The function create_vector makes the cost vector by forming a vector made up of different powers of the features. It was observed that the accuracy increased when the highest degree taken of the features was increased. Initially, the accuracy over aspell was less than 85%, but after increasing the degree, the accuracy increased to 91%.
## Contribute
Contributions and suggestions are welcome.
