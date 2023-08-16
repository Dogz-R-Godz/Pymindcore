import numpy as np

class neural_network:
    def __init__(self,inputs,middle_neurons,outputs):
        self.inputs=inputs #interger of number of inputs
        self.middle=middle_neurons #numpy array of the layers, eg: 
        self.outputs=outputs

    def initialise_matrices_2(self, weights):
        def create_matrices(num_neurons, weight_list):
            matrices = [[] for _ in range(num_neurons)]
            for idx, weight in enumerate(weight_list):matrices[idx % num_neurons].append(weight)
            return [np.array(matrix) for matrix in matrices]

        input_matrices = create_matrices(self.inputs, weights[0])

        middle_matrices = [create_matrices(self.middle[layer], weight_list) for layer, weight_list in enumerate(weights[1:])]

        self.weight_matrices = np.array([input_matrices] + middle_matrices)
            




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

            arr = np.empty(weights_needed, dtype = float)
            arr.fill(1/5000)
            weights.append(arr)
            biases.append(np.zeros(self.middle[0]))
            for m in range(len(self.middle)-1):
                biases.append(np.zeros(self.middle[m+1]))
                weights_needed=self.middle[m+1]*self.middle[m]
                arr = np.empty(weights_needed, dtype = float)
                arr.fill(1/5000)
                weights.append(arr)
            arr = np.empty(self.outputs*self.middle[-1], dtype = float)
            arr.fill(1/5000)
            weights.append(arr)
            biases.append(np.zeros(self.outputs))
        network=np.array([weights,biases])
        self.network=network
        self.initialise_matrices_2(weights)
        return network

    def activate(self,numbers,activation):
        if activation=="sig":return (1/(1+np.exp(-numbers)))
        if activation=="RELU":return np.ma.clip(numbers,0,numbers.max())
        if activation=="tanh":return np.tanh(numbers)
        else:raise Exception("Activation TypeError: Please enter a valid activation function. If you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")

    def get_output(self,inputs,activation):
        try:matrices=self.weight_matrices
        except:raise Exception("Weight Matrix Initialisation error: Please initialise the weight matrix with initialise_matrices(weights), or if you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")
        layer=0
        layers=self.middle.tolist()
        layers.insert(0,self.inputs)
        status=[inputs]
        unactivated_status=[inputs]
        while layer<(len(self.middle)+1):
            curr_layer=np.array([0.0])
            for neuron in range(layers[layer]):curr_layer=(matrices[layer][neuron]*status[layer][neuron]) + curr_layer

            status.append(self.activate(curr_layer,activation))
            unactivated_status.append(curr_layer)
            layer+=1
        x=0
        return status[-1], [status,unactivated_status]

    def find_error(self,expected_outputs,states,activation):
        total_error=0
        for state in range(len(states)):
            output=self.get_output(states[state],activation)
            total_error+=np.sum((output[0]-expected_outputs[state])**2)
        return total_error
    
    def error_deriv(self,outputs,expected_outputs):
        return 2*(outputs-expected_outputs)
    
    def activation_deriv(self,values,activation):
        if activation=="sig":return self.activate(values,activation)*(1-self.activate(values,activation)) 
        elif activation=="RELU":
            values2=values.copy()
            values2[values2 <= 0] = 0.01
            values2[values2 > 0] = 1
            return values2
        elif activation=="tanh":return 1-(np.tanh(values)**2)
        else:raise Exception("Activation TypeError: Please enter a valid activation function. If you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")

        


    def backpropergation(self,expected_outputs,states,activation):
        initial_error=self.find_error(expected_outputs,states,activation) #find initial error
        print(initial_error) #print initial error
        for state in range(len(states)): #loop through each state
            curr_state_error=self.find_error([expected_outputs[state]],[states[state]],activation)
            outputs,status=self.get_output(states[state],activation)
            
            #derivatives needed: 
            #1: error with respect to each output neuron
            #2: each output neuron with respect to the unactivated output neuron

            #3: each unactivated neuron in the network with respect to the prevous layer activated neuron (the weight connecting them)
            #4: each unactivated neuron in the network with respect to the weights connecting them to the prevous layer activated neuron (the prevous layer activated neuron)
            #5: each unactivated neuron in the network with respect to the bias (1)

            #6: multiply the output error deriv by the unactivated output error deriv to find the derivative for each output with respect to the unactivated neuron. TEST IT.

            output_neuron_error_derivs=self.error_deriv(outputs,expected_outputs[state]) #1
            print(output_neuron_error_derivs)
            unactivated_output_neuron_activation_derivs=self.activation_deriv(status[1][-1],activation) #2
            print(unactivated_output_neuron_activation_derivs)
            print(status[1][-1])
            print(" ")
            
            






if __name__=="__main__":
    network=neural_network(2,np.array([5,4]),2)
    Network=network.randomise_network()
    output=network.get_output(np.array([1,1]),"sig")
    states=np.array([np.array([0,0]),np.array([1,0]),np.array([0,1]),np.array([1,1])])
    expected_outputs=np.array([np.array([0,1]),np.array([1,0]),np.array([1,0]),np.array([0,1])])
    print(network.find_error(expected_outputs,states,"sig"))
    print(output)
    print(network.backpropergation(expected_outputs,states,"sig"))
