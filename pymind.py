import numpy as np
import time 
import math as maths
import math
import time
import numpy as np2
from PIL import Image
#from discord_webhook import DiscordWebhook

class Optomised_neural_network:
    def __init__(self, input_neurons, hidden_layers, output_neurons, activation="relu",random_init=True,tanh_output=False):
        self.layers = [input_neurons] + hidden_layers + [output_neurons]
        self.weights = []
        self.biases = []
        self.tanh_output=tanh_output
        
        if random_init:
            for i in range(len(self.layers) - 1):
                self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]))
                self.biases.append(np.random.randn(self.layers[i + 1]))
        else:
            for i in range(len(self.layers) - 1):
                self.weights.append(np.zeros((self.layers[i], self.layers[i + 1])))
                self.biases.append(np.zeros(self.layers[i + 1]))
        
        # Set activation function
        if activation == "relu":
            self.activation = self.relu
            self.activation_prime = self.relu_prime
        elif activation == "sig":
            self.activation = self.sigmoid
            self.activation_prime = self.sigmoid_prime
        elif activation == "tanh":
            self.activation = self.tanh
            self.activation_prime = self.tanh_prime

    def relu(self, x):
        return np.maximum(0.01*x, x)
    
    def relu_prime(self, x):
        return np.where(x > 0, 1, 0.01)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_prime(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)
    def tanh_prime(self, x):
        return 1 - np.tanh(x)**2

    def save_to_file(self, filename,isCupy=False):
        # Convert each CuPy array in the lists to a NumPy array
        if isCupy:
            weights_numpy = [weight.get() for weight in self.weights]
            biases_numpy = [bias.get() for bias in self.biases]
        else:
            weights_numpy = self.weights
            biases_numpy = self.biases

        # Now create the NumPy array to bundle the data
        data = np2.array([weights_numpy, biases_numpy], dtype=object)

        # Save the data using NumPy's save method
        np2.save(filename, data)
    
    def load_from_file(self, filename):
        # Load the data using NumPy's load method
        data = np2.load(filename, allow_pickle=True)

        # Convert the loaded NumPy arrays back to CuPy arrays and set to self.weights and self.biases
        self.weights = [np.array(arr) for arr in data[0]]
        self.biases = [np.array(arr) for arr in data[1]]
    
    def forward(self, x, tanh_output=False):
        self.a = [x]
        for i in range(len(self.weights) - 1):
            #curr_weights=self.weights[i].copy()
            #curr_weights=np.stack(curr_weights,self.a[-1].size)

            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.a.append(self.activation(z))
        z_out = self.a[-1] @ self.weights[-1] + self.biases[-1]
        if tanh_output:self.a.append(self.tanh(z_out))
        else:self.a.append(self.sigmoid(z_out))
        
        return self.a[-1]
    
    def compute_loss(self, y):
        m = y.shape[0]
        eachStateError=(self.a[-1] - y) ** 2
        return np.sum(eachStateError)/m #(1 / (2 * m)) * 

    def find_error(self, x, y):
        # Compute predictions for the given input states
        predictions = self.forward(x,self.tanh_output)
        # Compute the error for each sample and sum them up
        differences=(predictions - y)
        squared=np.sum(differences ** 2)
        total_error = squared/len(x)
        return total_error
    
    def backward(self, x, y, learning_rate=0.1, reduceDeriv=False):
        startingTime=time.time()
        m = x.shape[0]
        self.dz = [-(y - self.a[-1]) / m]
        for i in reversed(range(len(self.weights))):
            dw = self.a[i].T @ self.dz[-1]
            db = np.sum(self.dz[-1], axis=0)
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            if i != 0:
                if reduceDeriv:
                    act=self.activation_prime(self.a[i])
                    dz_next = ((self.dz[-1] @ self.weights[i].T)/(0.5*(10**i))) * act
                    self.dz.append(dz_next)
                else:
                    dz_next = self.dz[-1] @ self.weights[i].T * self.activation_prime(self.a[i])
                    self.dz.append(dz_next)
                
    def train(self, x, y, epochs=10000, learning_rate=0.01, optimizer=None, beta1=0.9, beta2=0.999, epsilon=1e-8, print_=False): #ping_webhook_URL=None,
        if optimizer == "ADAM":
            # Initialize Adam-specific variables
            m_weights = [np.zeros_like(w) for w in self.weights]
            v_weights = [np.zeros_like(w) for w in self.weights]
            m_biases = [np.zeros_like(b) for b in self.biases]
            v_biases = [np.zeros_like(b) for b in self.biases]
            t = 0
        totalTime=0
        prevError=99999999
        for epoch in range(epochs):
            startingTime=time.time()
            self.forward(x,self.tanh_output)
            self.backward(x, y, learning_rate,True)
            
            if optimizer == "ADAM":
                t += 1
                for i in range(len(self.weights)):
                    # Compute gradients for weights and biases
                    grad_w = self.a[i].T @ self.dz[-(i+1)]
                    grad_b = np.sum(self.dz[-(i+1)], axis=0)
                    
                    # Update first moment for weights and biases
                    m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * grad_w
                    m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * grad_b
                    
                    # Update second moment for weights and biases
                    v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * grad_w ** 2
                    v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * grad_b ** 2
                    
                    # Bias-corrected moments
                    m_weights_corr = m_weights[i] / (1 - beta1**t)
                    m_biases_corr = m_biases[i] / (1 - beta1**t)
                    v_weights_corr = v_weights[i] / (1 - beta2**t)
                    v_biases_corr = v_biases[i] / (1 - beta2**t)
                    
                    # Update weights and biases
                    self.weights[i] -= learning_rate * m_weights_corr / (np.sqrt(v_weights_corr) + epsilon)
                    self.biases[i] -= learning_rate * m_biases_corr / (np.sqrt(v_biases_corr) + epsilon)

                # Fix the clipping issue by iterating through each array in the lists
                for i in range(len(self.weights)):
                    self.weights[i] = np.clip(self.weights[i], -1, 1)
                    self.biases[i] = np.clip(self.biases[i], -1, 1)
            else:
                # Standard gradient descent
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * self.a[i].T @ self.dz[-(i+1)]
                    self.biases[i] -= learning_rate * np.sum(self.dz[-(i+1)], axis=0)
                for i in range(len(self.weights)):
                    self.weights[i] = np.clip(self.weights[i], -1, 1)
                    self.biases[i] = np.clip(self.biases[i], -1, 1)
            if epoch % math.ceil(epochs/100) == 0:
                
                self.forward(x)
                error=self.compute_loss(y)
                print(f"Epoch {epoch}/{epochs}. Error change: {prevError-error:.6f}. New error is {error:.6f}")
                prevError=error
                
                #if ping_webhook_URL != None:

                '''original_noise=np.random.uniform(0,255,self.layers[0])
                    
                    noise=original_noise.copy()
                    noise3=original_noise.copy()
                    noise3=noise3.reshape(50,50,3)
                    noise3=noise3.get()
                    original_noise=original_noise.get()
                    img = Image.fromarray(noise3.astype('uint8'), "RGB")  # Ensure the data type is uint8 for image
                    img.save("Trained_Images/starting_noise.png")
                    print(self.weights)
                    print(self.dz)
                    
                    for _ in range(500):
                        outputs=self.forward(noise/255)
                        
                        noise+=outputs
                        noise=np.clip(noise,0,255)
                        noise2=noise.reshape(50,50,3)

                        noise2_numpy = noise2.get()
                    img = Image.fromarray(noise2_numpy.astype('uint8'), "RGB")  # Ensure the data type is uint8 for image

                    img.save(f"Trained_Images/Epoch_{epoch}.png")'''

                    #webhook = DiscordWebhook(url=ping_webhook_URL, content=f'Done Milestone! {(epoch/epochs)*100}% done')
                '''with open(f"Trained_Images/Epoch_{epoch}.png", 'rb') as f:
                        webhook.add_file(file=f.read(), filename=f'Milestone_{epoch}_out_of_{epochs}.png')
                    with open(f"Trained_Images/starting_noise.png", 'rb') as f:
                        webhook.add_file(file=f.read(), filename=f'Starting Noise.png')'''
                    #response = webhook.execute()
                    
            if epochs>0:
                if epoch % math.ceil(epochs/5) == 0:
                    print(f"Epoch {epoch}/{epochs}")
                    self.save_to_file("Thing.npy")
            
            
            endingTime=time.time()
            totalTime+=endingTime-startingTime
            timeTaken=(totalTime/(epoch+1))*epochs
            print(f"Done epoch {epoch}/{epochs}. It will be done in {timeTaken-totalTime:.1f} seconds, or {(timeTaken-totalTime)/60:.1f} minutes, or {((timeTaken-totalTime)/60)/60:.2f} hours.")
            
            






#These below functions are depreciated.
def multiply_arrays_1d(array_of_arrays, numbers_array):
    if len(array_of_arrays) != len(numbers_array):
        return "Lengths of the arrays must be the same!"
    
    array_of_arrays = np.array(array_of_arrays)
    numbers_array = np.array(numbers_array)
    
    result = array_of_arrays * numbers_array[:, np.newaxis]
    return result.flatten()
class neural_network:
    def __init__(self,inputs,middle_neurons,outputs):
        print("This function is depretiated. Please use the Optomised_neural_network function. Support is coming soon for the DQL function again.")
        self.inputs=inputs 
        self.middle=middle_neurons 
        self.outputs=outputs

    def save_network(self,name="brain.npy"):
        np.save(name, self.network)
    def load_network(self,name="brain.npy"):
        self.network=np.load(name,allow_pickle=True)

    
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
        network=np.array([weights,biases],dtype=object)
        self.network=network
        self.initialise_matrices()
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
            for neuron in range(layers[layer]):
                curr_layer=(matrices[layer][neuron]*status[layer][neuron]) + curr_layer
            status.append(self.activate(curr_layer,activation))
            unactivated_status.append(curr_layer)
            layer+=1
        return status[-1], [status,unactivated_status]
    def better_get_output(self, inputs,activation):

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
    
    def chat_gippity_output(self, inputs, activation):
        try:weights=self.weight_matrices
        except:raise Exception("Weight Matrix Initialisation error: Please initialise the weight matrix with initialise_matrices(weights), or if you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")
        try:biases = self.network[1]
        except:raise Exception("Biases Initialisation error: Network biases are not defined, to define biases load or randomise the neural network, or if you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")

        last_vals = inputs
        all_vals = [inputs]
        unactivated_vals = [inputs]

        for layer in range(len(weights)):
            r3weights = weights[layer]#np.rot90(weights[layer], 1)
            #last_vals = np.rot90(last_vals, 1)
            layer_output = last_vals @ r3weights  # Use matrix multiplication
            layer_sum = np.flip(biases[layer] + layer_output)
            last_vals = self.activate(layer_sum, activation)
            all_vals.append(last_vals)
            unactivated_vals.append(layer_sum)

        output = last_vals
        return output, [all_vals, unactivated_vals]

    def find_error(self,expected_outputs,states,activation):
        total_error=0
        startingTime=time.time()
        for state in range(len(states)):
            output=self.chat_gippity_output(states[state],activation)
            total_error+=np.sum((output[0]-expected_outputs[state])**2)
            if time.time()-startingTime>=5:
                print(f"5 seconds have gone by, we are at state {state}")
                startingTime=time.time()
        return total_error/len(states)
    def error_deriv(self,outputs,expected_outputs):
        return 2*(outputs-expected_outputs)
    def activation_deriv(self,values,activation):
        if activation=="sig":return self.activate(values,activation)@(1-self.activate(values,activation)) 
        elif activation=="RELU":
            values2=values.copy()
            values2[values2 <= 0] = 0.01
            values2[values2 > 0] = 1
            return values2
        elif activation=="tanh":return 1-(np.tanh(values)**2)
        else:raise Exception("Activation TypeError: Please enter a valid activation function. If you're not a developer, dm dogzrgodz or That Guy#6482 on discord. ")
    def backpropergation_old(self,expected_outputs,states,activation):
        weights=0 #format this correctly
        biases=self.network[1].copy()
        biases = np.zeros_like(biases)
        weight_derivs_2=weights
        
        bias_derivs_2=biases
        state_time=time.time()
        for state in range(len(states)): 
            
            curr_state_error=self.find_error([expected_outputs[state]],[states[state]],activation)
            outputs,status=self.chat_gippity_output(states[state],activation)
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
                error_deriv_w_resp_unactiv = last_layer_error_derivs @ unactivated_curr_layer_activation_derivs
                #This part of the code prepared a delicous curry...
                unactivated_neuron_w_resp_weights = np.tile(prevous_layer, len(curr_layer))
                unactivated_neuron_w_resp_weights_2=np.split(unactivated_neuron_w_resp_weights,len(curr_layer))
                unactivated_neuron_w_resp_bias = np.ones(len(curr_layer))
                unactivated_neuron_w_resp_bias_2=np.split(unactivated_neuron_w_resp_bias,len(curr_layer))
                multiplied_deriv_with_weights=multiply_arrays_1d(unactivated_neuron_w_resp_weights_2,error_deriv_w_resp_unactiv)
                multiplied_deriv_with_bias=multiply_arrays_1d(unactivated_neuron_w_resp_bias_2,error_deriv_w_resp_unactiv)
                weight_derivs.insert(0,multiplied_deriv_with_weights)
                bias_derivs.insert(0,multiplied_deriv_with_bias)
                needed_arr=np.sum(curr_layer_weights_arr@last_layer_error_derivs,axis=1)
                size=np.shape(curr_layer_weights_arr)
                prev_layer_size=size[1]
                needed_arr/=prev_layer_size
                last_layer_error_derivs=needed_arr@unactivated_prevous_layer_activation_derivs
                #Enjoy the curry
            print(f"Done state {state}/{len(states)}. Done this in {time.time()-state_time} seconds")
            state_time=time.time()
            weight_derivs=np.array(weight_derivs)
            bias_derivs=np.array(bias_derivs)
            weight_derivs_2_2=[]
            size=np.shape(weight_derivs)
            for thing in range(size[0]):
                weight_derivs_2_2.append(weight_derivs[thing]@curr_state_error)
            
            weight_derivs_2_2=np.array(weight_derivs_2_2)
            if type(weight_derivs_2)==int:
                weight_derivs_2=weight_derivs_2_2
            else:
                weight_derivs_2+=weight_derivs_2_2
            bias_derivs_2+=bias_derivs@curr_state_error
        return weight_derivs_2/len(states), bias_derivs_2/len(states)

    def backpropergation(self,expected_outputs,states,activation):
        weights=0 #format this correctly
        biases=self.network[1].copy()
        biases = np.zeros_like(biases)
        weight_derivs_2=weights
        
        bias_derivs_2=biases
        for state in range(len(states)): 
            
            curr_state_error=self.find_error([expected_outputs[state]],[states[state]],activation)
            outputs,status=self.chat_gippity_output(states[state],activation)
            last_layer_error_derivs=self.error_deriv(outputs,expected_outputs[state])
            bias_derivs=[]
            weight_derivs=[]
            for layer in range(len(status[1])-1): 
                curr_layer=status[1][-(layer+1)]
                prevous_layer=status[1][-(layer+2)]
                curr_layer_weights=self.weight_matrices[-(layer+1)]
                curr_layer_weights_arr=np.array(curr_layer_weights)
                #This part of the code prepared a delicous curry...
                unactivated_curr_layer_activation_derivs=self.activation_deriv(curr_layer,activation)
                unactivated_prevous_layer_activation_derivs=self.activation_deriv(prevous_layer,activation)
                error_deriv_w_resp_unactiv = last_layer_error_derivs * unactivated_curr_layer_activation_derivs
                unactivated_neuron_w_resp_bias_2 = np.ones((len(curr_layer)))
                unactivated_neuron_w_resp_weights_2 = np.broadcast_to(prevous_layer, (len(curr_layer), len(prevous_layer)))
                reshaped_error_deriv = error_deriv_w_resp_unactiv[:, np.newaxis]
                multiplied_deriv_with_weights = unactivated_neuron_w_resp_weights_2 * reshaped_error_deriv
                multiplied_deriv_with_bias = unactivated_neuron_w_resp_bias_2 * error_deriv_w_resp_unactiv
                weight_derivs.insert(0, multiplied_deriv_with_weights)
                bias_derivs.insert(0, multiplied_deriv_with_bias)
                size = curr_layer_weights_arr.shape
                prev_layer_size = size[1]
                needed_arr = np.sum(curr_layer_weights_arr * last_layer_error_derivs, axis=1) / prev_layer_size
                last_layer_error_derivs = needed_arr * unactivated_prevous_layer_activation_derivs

            print(f"Done state {state}/{len(states)}.")

            weight_derivs=np.array(weight_derivs,dtype=object)
            bias_derivs=np.array(bias_derivs,dtype=object)
            weight_derivs_2_2=[]
            size=np.shape(weight_derivs)
            for thing in range(size[0]):
                weight_derivs_2_2.append(weight_derivs[thing]*curr_state_error)
            
            weight_derivs_2_2=np.array(weight_derivs_2_2,dtype=object)
            if type(weight_derivs_2)==int:
                weight_derivs_2=weight_derivs_2_2
            else:
                weight_derivs_2+=weight_derivs_2_2
            bias_derivs_2+=bias_derivs*min(curr_state_error,self.outputs)
        weight_derivs_2=weight_derivs_2/len(states)
        bias_derivs_2=bias_derivs_2/len(states)
        #bias_derivs_2 = bias_derivs_2[:, 0] #get the first number from each array
        return weight_derivs_2, bias_derivs_2
    def train_AI(self,epochs,expected_outputs,states,activation,learning_rate,optomiser=None,print_=False,strength_range=1):
        """Optomiser can be ADAM or None"""
        momentum=np.array([]) #momentum = momentum*0.99 + gradients*0.2
        if print_:
            print("About to find the initial error")
        initial_error=self.find_error(expected_outputs,states,activation)
        if print_:
            print("Found the initial error")
        prevous_error=initial_error
        for epoch in range(epochs):
            epoch_time=time.time()
            new_network=self.network.copy()
            new_network_2=self.network.copy()
            if print_:
                print("About to find the gradients")
            grads_weights,grads_biases=self.backpropergation(expected_outputs,states,activation)
            if print_:
                print("Found the gradients")
            
            if len(momentum)==0:
                weight_momentum=grads_weights
                bias_momentum=grads_biases
            if print_:
                print("About to start the optomiser")
            if optomiser=="ADAM":
                weight_momentum = weight_momentum*0.5 + grads_weights*0.5
                bias_momentum = bias_momentum*0.5 + grads_biases*0.5

                lrGradientsWei=learning_rate*weight_momentum
                lrGradientsBia=learning_rate*bias_momentum
                curr_weights=new_network[0].copy()
                new_weights=[]
                for layer in range(len(curr_weights)):
                    new_weights.append(np.clip(curr_weights[layer]-lrGradientsWei[layer],-strength_range,strength_range))

                curr_biases=new_network[1].copy()
                new_biases=[]
                for layer in range(len(curr_biases)):
                    new_biases.append(np.clip(curr_biases[layer]-lrGradientsBia[layer],-strength_range,strength_range))
                new_network=np.array([new_weights,new_biases],dtype=object) 
                self.network=new_network
                self.initialise_matrices()
            else:
                lrGradientsWei=learning_rate*grads_weights
                lrGradientsBia=learning_rate*grads_biases
                curr_weights=new_network[0].copy()
                new_weights=[]
                new_weights_2=[]
                for layer in range(len(curr_weights)):
                    new_lr_grad=np.rot90(lrGradientsWei[layer])
                    new_lr_grad1=np.rot90(new_lr_grad)
                    new_lr_grad1=np.rot90(new_lr_grad1)
                    new_weights.append(np.clip(curr_weights[layer]-new_lr_grad.flatten(),-strength_range,strength_range)) 
                    new_weights_2.append(np.clip(curr_weights[layer]-new_lr_grad1.flatten(),-strength_range,strength_range)) 

                curr_biases=new_network[1].copy()
                new_biases=[]
                for layer in range(len(curr_biases)):
                    new_biases.append(np.clip(curr_biases[layer]-lrGradientsBia[layer],-strength_range,strength_range))
                new_network=np.array([new_weights,new_biases],dtype=object) 
                new_network_2=np.array([new_weights_2,new_biases],dtype=object) 
                self.network=new_network
                self.initialise_matrices()
            print("Done the optomiser")
            if print_:
                if epoch%maths.ceil(epochs/100)==0 or time.time()-epoch_time>10: #if its time to update, or its been over 10 seconds since the last epoch update
                    recent_error=self.find_error(expected_outputs,states,activation)
                    print(f"Epoch {epoch}/{epochs} done. Initial error: {initial_error}, Last error: {prevous_error} New error: {recent_error}. ")

                    self.network=new_network_2
                    self.initialise_matrices()
                    recent_error_2=self.find_error(expected_outputs,states,activation)
                    print(f"Epoch {epoch}/{epochs} done. Initial error: {initial_error}, Last error: {prevous_error} New error: {recent_error_2}. For test network")
                    if recent_error>recent_error_2:
                        print("recent_error_2 is smaller")
                        prevous_error=recent_error_2
                        self.network=new_network_2
                        self.initialise_matrices()
                        

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
    def get_next_frame(self,state):
        #add to replay buffer
        output=self.nn.get_output(state,self.activation)
        first_index = int(np.where(output[0] == output[0].max())[0][0])
        return first_index, output
    def better_get_next_frame(self, state):
        output = self.nn.chat_gippity_output(state, self.activation)
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
        self.replay_buffer=[]
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
