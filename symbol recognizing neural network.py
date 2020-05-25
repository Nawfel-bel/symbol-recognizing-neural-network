import numpy as np
import scipy.special


class neuralNetwork:

    def __init__(self, inputnodes, outputnodes, hiddennodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.wih = np.random
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass


    def train (self,inputs_list,targetslist):
        inputs = np.array(inputs_list,ndmin =2).T
        targets = np.array(targetslist,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_error = targets- final_outputs
        hidden_errors = np.dot(self.who.T,output_error)
        self.who += self.lr * np.dot((output_error*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr *np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
        pass

    def query(self, inputsls):
        inputs = np.array(inputsls, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

