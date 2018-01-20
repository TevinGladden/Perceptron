# Language:       python2
# Creator:        Tevin Gladden
# Project:        Perceptron-Email spam binary classifier
#
# This single Perceptron uses a data set of 15497 feature vectors, each with 335 features.
# Each instance is a representation of an email, where feature 0 is the classification(spam or non-spam)
# The remaining 334 features represent specific qualities of the email in question.
# 75 percent of the data set goes into a training set and the remaining 25 percent to a test set.
# The Perceptron is set to train until it's prediction accuracy reaches 98 percent or 1000 epochs.

import numpy as np

class Perceptron:
    def __init__(self):
        self.trainingSet = []
        self.validationSet = []
        self.weights = []
        self.accuracy = 0.0
        self.learningRate = .0170000999
        self.epoch = 1
        self.finalWeights = []
        
    def inputs(self):
        counter = 0
        train = open("training.txt", mode='r')
        validate = open("validation.txt", mode='r')
        
        for line in train:
            self.trainingSet.append(line.strip("\n").split(" "))
            self.trainingSet[counter][:-1] = map(lambda x: int(x), self.trainingSet[counter][:-1])
            self.trainingSet[counter][-1] = np.array([int(digit) for digit in self.trainingSet[counter][-1]])
            counter += 1
        
        counter = 0
         
        for line in validate:
            self.validationSet.append(line.strip("\n").split(" "))
            self.validationSet[counter][:-1] = map(lambda x: int(x), self.validationSet[counter][:-1]) 
            self.validationSet[counter][-1] = np.array([int(digit) for digit in self.validationSet[counter][-1]])
            counter += 1
        
        self.weights = 2*np.random.random(334)-1
        self.final = np.append(np.copy(self.weights), 0.0)
        
    def train(self):
        prediction = 0
        for i in range(len(self.trainingSet)):
            prediction = (self.trainingSet[i][-1].dot(self.weights))+1
            if prediction <= 0 and self.trainingSet[i][1] == 1 or prediction > 0 and self.trainingSet[i][1] == -1:
                self.weights += np.multiply(self.learningRate*(self.trainingSet[i][1]-prediction), self.trainingSet[i][2])
           
    def test(self):
        correct = 0.0
        for i in range(len(self.validationSet)):
            prediction = (self.validationSet[i][-1].dot(self.weights))
            if prediction > 0 and self.validationSet[i][1] == 1 or prediction <= 0 and self.validationSet[i][1] == -1:
                correct = correct + 1
            
        self.accuracy = correct / len(self.validationSet)
        
    def run(self):
        self.inputs()
        while self.accuracy < .98 and self.epoch <> 1000:
            self.accuracy = 0    
            self.train()
            self.test()
            print "Epoch: "+str(self.epoch)
            print "Validation Error: "+str(1-self.accuracy)+"\n"
            self.epoch += 1
            if(self.accuracy > self.final[-1]):
                self.final[:-1] = np.copy(self.weights)
                self.final[-1] = self.accuracy
        print("\nBest Accuracy: "+str(self.final[-1]))
            
Perceptron().run()