import numpy as np

class NeuralNetwork:
    def __init__(self,L,hiddenNeurons):
        self.L = L
        self.neuron = hiddenNeurons       
        
    def forwardPropagate(self):
        self.O = [self.inputLayer]
        net = self.activate(np.dot(self.inputLayer,(self.W[0])))
        self.O.append(net)
        for i in range(1,(self.L)-1):
            net = self.activate(np.dot(net,(self.W[i])))
            self.O.append(net)
        return net
    
    def activate(self,x,deriv=False):
        if (not deriv):
            return 1.0/(1.0+np.exp(-x))
        else:
            return self.activate(x)*(1-self.activate(x))
    
    def declareWeights(self):
        #np.random.seed(111)
        self.W = []
        weight = 2*np.random.random((self.inputs,self.neuron)) - 1
        self.W.append(weight)
        for i in range(1,(self.L)-2):
            weight = 2*np.random.random((self.neuron,self.neuron)) - 1
            self.W.append(weight)
        weight = 2*np.random.random((self.neuron,1)) - 1
        self.W.append(weight)

    def talk(self,corpus):
        self.engine.say(corpus)
        self.engine.runAndWait()
        
    def normalize(inputLayer):
        pass

    def declareInput(self,inputLayer):
        self.inputLayer = inputLayer
        self.dataPoints = inputLayer.shape[0]
        self.inputs = inputLayer.shape[1]
    
    def declareTarget(self,targetLayer):
        self.targetLayer = targetLayer
        self.targets = targetLayer.shape[0]
        self.declareWeights()

    def backPropagate(self,output):
        error = self.targetLayer - output
        #errorPercent = self.getErrorPercent(output)
        errormean = np.mean(np.abs(error))
        delta = error*self.O[-1]*(1-self.O[-1])
        for i in range(1,self.L):
            error = delta.dot(self.W[-i].T)
            self.W[-i]+=0.2*self.O[-i-1].T.dot(delta)
            delta = error*self.O[-i-1]*(1-self.O[-i-1])
        return errormean
    def getWeights(self):
        return self.W

    def train(self,epochs):
        for i in range(0,epochs):
            output = self.forwardPropagate()
            error = self.backPropagate(output)
            if i%1000 == 0:
                print "Error after " + str(i)  + " epochs: " + str(error)
                
    def getL(self):
        return self.O

    def Test(self):
        pass
    def getErrorPercent(self,output):
        o = output*(3.093-0.223)+0.223
        a = self.targetLayer*(3.093-0.223)+0.223
        error = np.mean(np.abs(((o-a)/a)))*100
        return error
