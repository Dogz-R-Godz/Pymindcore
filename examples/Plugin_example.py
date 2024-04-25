import pymindcore as pm
import numpy as np
from pymindcore import plugins
from time import perf_counter


nn=pm.Optomised_neural_network(4,[4,4],2,"relu","relu","mean")

weights=nn.weights.copy()
data=np.array([[0,0,
               0,0],
              [1,0,
               0,0],
              [0,1,
               0,0],
              [1,1,
               0,0],
              [0,0,
               1,0],
              [1,0,
               1,0],
              [0,1,
               1,0],
              [1,1,
               1,0],
              [0,0,
               0,1],
              [1,0,
               0,1],
              [0,1,
               0,1],
              [1,1,
               0,1],
              [0,0,
               1,1],
              [1,0,
               1,1],
              [0,1,
               1,1],
              [1,1,
               1,1]])
expected_outputs=np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0],[0,1],[0,1],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])

nn=plugins.injectPlugins(nn)
start=perf_counter()
nn.train(data.astype(np.float64), expected_outputs.astype(np.float64),20000,0.001,"ADAM",print_=False)
end=perf_counter()
loss=nn.find_error(data.astype(np.float64), expected_outputs.astype(np.float64), False)
print(f"Training took {end-start} seconds. Final error is {loss}")
nn.a=[data.astype(np.float64)]
nn.forward(None)
print(nn.a)

nn.save_to_file("Network out.npy")
