import numpy as np
import numexpr as ne

def pluginInit(self, settings):
    print(settings)
    self.compute_loss=self.compute_loss2f
    self.find_error=self.find_error2f


def compute_loss2f(self, y):
    m = y.shape[0]
    eachStateError=ne.evaluate("(a - y) ** 2", {'a': self.a[-1], 'y': y})
    return np.sum(eachStateError)/m #(1 / (2 * m)) * 

def find_error2f(self, x, y, verbose=False):
    # Compute predictions for the given input states
    self.a=[x]
    predictions = self.forward(None,verbose) #x
    if verbose:print("Done forward pass")
    # Compute the error for each sample and sum them up
    differences=(predictions - y)
    if verbose:print("Found differences")
    squared=np.sum(differences ** 2)
    if verbose:print("Squared all the differences")
    total_error = squared/len(x)
    if verbose:print("Found total error")
    return total_error

"""def sig_scalingf(self, x, speedy=True):
    a=self.pll
    w=self.clip_amount
    h=0.9999
    if speedy: return ne.evaluate("(1 / (1+exp(-(x*((-log((1/h)-1)/w)/a*0.5)))))")
    return (1 / (1+np.exp(-(x*((-np.log((1/h)-1)/w)/a*0.5)))))

def inv_sig_scalingf(self, x, speedy=True):
    x=np.clip(x, 0.0001, 0.9999)
    a=self.pll
    w=self.clip_amount
    h=0.9999
    if speedy: return ne.evaluate("-log((1/x)-1) / ((-log((1/h)-1)/w)/(a*0.5))")
    return -np.log((1/x)-1) / ((-np.log((1/h)-1)/w)/(a*0.5))

def sig_scaling_primef(self, x, speedy=True):
    x=np.clip(x, 0.0001, 0.9999)
    a=self.pll
    w=self.clip_amount
    h=0.9999
    if speedy: return ne.evaluate("(-(2*log((1/h)-1)*exp(((2*log((1/h)-1)*x)/(a*w)))) / (a*w*((exp(((2*log((1/h)-1)*x)/(a*w)))+1)**2)))")
    return (-(2*np.log((1/h)-1)*np.exp(((2*np.log((1/h)-1)*x)/(a*w)))) / (a*w*((np.exp(((2*np.log((1/h)-1)*x)/(a*w)))+1)**2)))
    """