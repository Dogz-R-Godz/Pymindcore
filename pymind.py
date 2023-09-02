import numpy as np
import time 
import math as maths
def multiply_arrays_1d(array_of_arrays, numbers_array):
    if len(array_of_arrays) != len(numbers_array):
        return "Lengths of the arrays must be the same!"
    
    array_of_arrays = np.array(array_of_arrays)
    numbers_array = np.array(numbers_array)
    
    result = array_of_arrays * numbers_array[:, np.newaxis]
    return result.flatten()
class neural_network:
    def __init__(self,inputs,middle_neurons,outputs):
        self.inputs=inputs 
        self.middle=middle_neurons 
        self.outputs=outputs

    def save_network(self,name="brain.npy"):
        np.save(name, self.network)
    def load_network(self,name="brain.npy"):
        self.network=np.load(name)

    
    def initialise_matrices(self):
        weights = self.network[0]

        def create_matrices(num_neurons, weight_list):
            # Reshape the weight list to form the matrices
            return np.array(weight_list).reshape((-1, num_neurons)).T

        input_matrices = create_matrices(self.inputs, weights[0])
        middle_matrices = [create_matrices(self.middle[layer], weight_list) 
                           for layer, weight_list in enumerate(weights[1:])]
        
        # Use a list to store matrices since they can have different shapes
        self.weight_matrices = [input_matrices] + middle_matrices
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
            arr.fill(0)
            weights.append(arr)
            biases.append(np.zeros(self.middle[0]))
            for m in range(len(self.middle)-1):
                biases.append(np.zeros(self.middle[m+1]))
                weights_needed=self.middle[m+1]*self.middle[m]
                arr = np.empty(weights_needed, dtype = float)
                arr.fill(0)
                weights.append(arr)
            arr = np.empty(self.outputs*self.middle[-1], dtype = float)
            arr.fill(0)
            weights.append(arr)
            biases.append(np.zeros(self.outputs))
        maxsize = max(max([[s.size for s in weights],[s.size for s in biases]]))
        #weights = [np.pad(s, (0,maxsize-s.size), 'constant', constant_values=0 ) for s in weights]
        #biases = [np.pad(s, (0,maxsize-s.size), 'constant', constant_values=0) for s in biases]
        network=np.array([weights,biases])
        self.network=network
        self.initialise_matrices()
    def activate(self,numbers,activation):
        if activation=="sig":return (1/(1+np.exp(-numbers)))
        if activation=="RELU":return np.ma.clip(numbers,0,numbers.max())
        if activation=="tanh":return np.tanh(numbers)
        else:raise Exception("Activation TypeError: Please enter a valid activation function. If you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")
    def get_output(self, inputs,activation):
        try:weights=self.weight_matrices
        except:raise Exception("Weight Matrix Initialisation error: Please initialise the weight matrix with initialise_matrices(weights), or if you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")
        try:biases = self.network[1]
        except:raise Exception("Biases Initialisation error: Network biases are not defined, to define biases load or randomise the neural network, or if you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")
        last_vals = inputs
        all_vals = [inputs]
        unactivated_vals = [inputs]
        for layer in range(len(weights)):
            r3weights = np.rot90(weights[layer],1)
            layer_output = np.rot90(last_vals * r3weights,3)
            layer_sum = biases[layer] + np.sum(layer_output,axis=0)
            last_vals = self.activate(layer_sum,activation)
            all_vals.append(last_vals)
            unactivated_vals.append(layer_sum)
        output = last_vals
        return output, [all_vals,unactivated_vals]
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
        weights=self.weight_matrices.copy()
        weights=np.zeros_like(weights)
        biases=self.network[1].copy()
        biases = np.zeros_like(biases)
        weight_derivs_2=weights
        
        bias_derivs_2=biases
        for state in range(len(states)): 
            state_time=time.time()
            curr_state_error=self.find_error([expected_outputs[state]],[states[state]],activation)
            outputs,status=self.get_output(states[state],activation)
            last_layer_error_derivs=self.error_deriv(outputs,expected_outputs[state])
            bias_derivs=[]
            weight_derivs=[]
            for layer in range(len(status[1])-1): 
                curr_layer=status[1][-(layer+1)]
                prevous_layer=status[1][-(layer+2)]
                curr_layer_weights=self.weight_matrices[-(layer+1)]
                curr_layer_weights_arr=np.array(curr_layer_weights)
                unactivated_curr_layer_activation_derivs=self.activation_deriv(curr_layer,activation)
                unactivated_prevous_layer_activation_derivs=self.activation_deriv(prevous_layer,activation)
                error_deriv_w_resp_unactiv = last_layer_error_derivs * unactivated_curr_layer_activation_derivs
                #This part of the code prepared a delicous curry...
                unactivated_neuron_w_resp_weights = np.tile(prevous_layer, len(curr_layer))
                unactivated_neuron_w_resp_weights_2=np.split(unactivated_neuron_w_resp_weights,len(curr_layer))
                unactivated_neuron_w_resp_bias = np.ones(len(curr_layer))
                unactivated_neuron_w_resp_bias_2=np.split(unactivated_neuron_w_resp_bias,len(curr_layer))
                multiplied_deriv_with_weights=multiply_arrays_1d(unactivated_neuron_w_resp_weights_2,error_deriv_w_resp_unactiv)
                multiplied_deriv_with_bias=multiply_arrays_1d(unactivated_neuron_w_resp_bias_2,error_deriv_w_resp_unactiv)
                weight_derivs.insert(0,multiplied_deriv_with_weights)
                bias_derivs.insert(0,multiplied_deriv_with_bias)
                needed_arr=np.sum(curr_layer_weights_arr*last_layer_error_derivs,axis=1)
                last_layer_error_derivs=needed_arr*unactivated_prevous_layer_activation_derivs
                #Enjoy the curry
            if time.time()-state_time>0.5:
                print(f"Done state {state}/{len(states)}")
            weight_derivs=np.array(weight_derivs)
            bias_derivs=np.array(bias_derivs)
            weight_derivs_2+=weight_derivs*curr_state_error
            bias_derivs_2+=bias_derivs*curr_state_error
        network_derivs=np.array([weight_derivs_2/len(states),bias_derivs_2/len(states)])
        return network_derivs
    def train_AI(self,epochs,expected_outputs,states,activation,learning_rate,optomiser=None,print_=False):
        """Optomiser can be ADAM or None"""
        momentum=np.array([]) #momentum = momentum*0.99 + gradients*0.2
        for epoch in range(epochs):
            epoch_time=time.time()
            new_network=self.network.copy()
            gradients=self.backpropergation(expected_outputs,states,activation)
            if len(momentum)==0:
                momentum=gradients
            if optomiser=="ADAM":
                momentum = momentum*0.5 + gradients*0.5
                lrGradients=learning_rate*momentum
                new_network=new_network-lrGradients
                self.network=new_network
                self.initialise_matrices()
            else:
                lrGradients=learning_rate*gradients
                print(lrGradients[0][0].max())
                new_network=new_network-lrGradients
                self.network=new_network
                self.initialise_matrices()
            if print_:
                if epoch%maths.ceil(epochs/100)==0 or time.time()-epoch_time>10: #if its time to update, or its been over 10 seconds since the last epoch update
                    print(f"Epoch {epoch}/{epochs} done. New error: {self.find_error(expected_outputs,states,activation)}. ")

class DQL:
    def __init__(self,inputs,middle,outputs,activation,learning_rate):
        self.inputs=inputs
        self.middle=middle
        self.outputs=outputs
        self.activation=activation
        self.learning_rate=learning_rate
        self.replay_buffer=[] #[state, action, outputs, reward, terminal]
        self.nn=neural_network(inputs,middle,outputs)
    def save_network(self,name="brain.npy"):
        np.save(name, self.nn.network)
    def load_network(self,name="brain.npy"):
        self.nn.network=np.load(name,allow_pickle=True)
        self.nn.initialise_matrices()
    def randomise_brain(self):
        self.nn.randomise_network()
    def get_next_frame(self, state):
        output = self.nn.get_output(state, self.activation)
        first_index = int(np.where(output[0] == output[0].max())[0][0])
        return first_index, output
    def replay_buffer_adder(self,state,action,outputs,reward,terminal):
        self.replay_buffer.append([state,action,outputs,reward,terminal])
    def prepare_for_backprop(self,reward_decay=0.99,randomise=True):
        expected_outputs=[]
        states=[]
        rev_replay_buffer=self.replay_buffer.copy()
        rev_replay_buffer.reverse()
        prev_replay_reward=0
        for replay in rev_replay_buffer: #[state, action, outputs, reward, terminal]
            states.append(replay[0].copy())
            action=replay[1]
            outputs=replay[2][1][0].copy()
            reward=replay[3] + prev_replay_reward*reward_decay
            terminal=replay[4]
            if terminal:
                prev_replay_reward=0
            else:
                prev_replay_reward=reward
            outputs[action]=reward
            expected_outputs.append(outputs)
        if randomise:
            len_states=len(states)
            states_2=[]
            expected_outputs_2=[]
            all_nums=np.random.randint(0,len_states,len_states)
            for state in range(len_states):
                states_2.append(states[all_nums[state]])
                expected_outputs_2.append(expected_outputs[all_nums[state]])
            return expected_outputs_2, states_2

        return expected_outputs, states
    def complete_session(self,expected_outputs,states,epochs,optomiser=None,print_=False):
        self.nn.train_AI(epochs,expected_outputs,states,self.activation,self.learning_rate,optomiser,print_)
        if print_:
            print("Training session complete")
    def save(self):
        self.nn.save_network()
    def load(self):
        self.nn.load_network()






class RNN:
    def __init__(self,inputs,middle,outputs):
        self.inputs=inputs
        self.middle=middle
        self.outputs=outputs
        thing=[np.array([np.zeros(self.middle[layer])]) for layer in range(len(self.middle))]
        self.prev_middle_neurons=[thing]

    def randomise_network(self,random=True):
        x=0
        network=[]
        if random:
            prevous_state_bias=np.array([np.random.uniform(0.9,1.1,self.middle[layer]) for layer in range(len(self.middle))])

#network=RNN(3,np.array([5,4,6]),2)
#network.randomise_network()
