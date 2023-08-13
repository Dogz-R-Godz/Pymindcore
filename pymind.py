import numpy as np

class neural_network:
    def __init__(self,inputs,middle_neurons,outputs):
        self.inputs=inputs #interger of number of inputs
        self.middle=middle_neurons #numpy array of the layers, eg: 
        self.outputs=outputs

    def initialise_matrices(self,weights):
        curr_input=0
        full_matrices=[]
        input_matrices=[]
        for _ in range(self.inputs):
            input_matrices.append(np.array([]))
        for weight in weights[0]: #loop through each weight in the first layer of weights
            input_matrices[curr_input]=np.append(input_matrices[curr_input],weight)
            curr_input+=1
            if curr_input==self.inputs:
                curr_input=0
        #add middle neuron support
        full_matrices=[input_matrices]
        for weight_layer in range(len(weights)-1): #loop through the weight layers minus one (input layers)
            curr_layer=[]
            curr_middle=0
            for _ in range(self.middle[weight_layer]):
                curr_layer.append(np.array([]))
            for weight in weights[weight_layer+1]: #loop through each weight in that layer
                curr_layer[curr_middle]=np.append(curr_layer[curr_middle],weight)
                curr_middle+=1
                if curr_middle==self.middle[weight_layer]:
                    curr_middle=0
            full_matrices.append(curr_layer)
        self.weight_matrices=np.array(full_matrices)
            
        x=0




    def randomise_network(self,weight_range=1,random=True):
        weights=[]
        biases=[]
        if random:
            weights_needed=self.middle[-1]*self.outputs
            all_weights=np.random.uniform(-weight_range,weight_range,self.inputs*self.middle[0])
            weights.append(all_weights)
            
            biases.append(np.zeros(self.middle[0]))
            for m in range(len(self.middle)-1):
                biases.append(np.zeros(self.middle[m+1]))

                weights_needed=(self.middle[m+1]*self.middle[m])
                all_weights=np.random.uniform(-weight_range,weight_range,weights_needed)
                weights.append(all_weights)
            all_weights=np.random.uniform(-weight_range,weight_range,self.middle[-1]*self.outputs)
            weights.append(all_weights)

            biases.append(np.zeros(self.outputs))
        else:
            weights_needed=self.inputs*self.middle[0]
            weights.append(np.zeros(weights_needed))
            biases.append(np.zeros(self.middle[0]))
            for m in range(len(self.middle)-1):
                biases.append(np.zeros(self.middle[m+1]))
                weights_needed=self.middle[m+1]*self.middle[m]
                weights.append(np.zeros(weights_needed))
            weights.append(np.zeros(self.outputs*self.middle[-1]))
            biases.append(np.zeros(self.outputs))
        network=np.array([np.array(weights),np.array(biases)],dtype=object)
        self.network=network
        self.initialise_matrices(weights)
        return network

    def get_output(self,inputs,activation):
        try:
            network=self.network
        except:
            raise Exception("Network Initialisation error: Please initialise the network with randomise_network(weight_range,random), or initialise_network(network) before you call get_output")

        layers=inputs
        layers=np.append(layers,self.middle)




network=neural_network(2,np.array([5,4]),3)
Network=network.randomise_network()