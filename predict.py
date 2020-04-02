import numpy as np
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
ip = np.array([[0,0],[0,1],[1,0],[1,1]])
op = np.array([[0],[1],[1],[0]])
epochs = 10000
alpha = 0.1
num_ip_layer = 2
num_h_layer = 2
num_o_layer = 1
h_w = np.random.uniform(size=(num_ip_layer,num_h_layer))
h_b =np.random.uniform(size=(1,num_h_layer))
o_w = np.random.uniform(size=(num_h_layer,num_o_layer))
o_b = np.random.uniform(size=(1,num_o_layer))

for _ in range(epochs):
	#Forward Propagation
	h_layer_actiavtion = np.dot(ip,h_w)
	h_layer_actiavtion += h_b
	op_layer = sigmoid(h_layer_actiavtion)

	op_activation = np.dot(op_layer,o_w)
	op_activation += o_b
	nn_output = sigmoid(op_activation)

	#Backpropagation
	error = op - nn_output
	d_nn_output = error * sigmoid_derivative(nn_output)
	
	error_num_h_layer = d_nn_output.dot(o_w.T)
	d_num_h_layer = error_num_h_layer * sigmoid_derivative(op_layer)

	o_w += op_layer.T.dot(d_nn_output) * alpha
	o_b += np.sum(d_nn_output,axis=0,keepdims=True) * alpha
	h_w += ip.T.dot(d_num_h_layer) * alpha
	h_b += np.sum(d_num_h_layer,axis=0,keepdims=True) * alpha
print("Final hidden weights: ",end='')
print(*h_w)
print("Final hidden bias: ",end='')
print(*h_b)
print("Final output weights: ",end='')
print(*o_w)
print("Final output bias: ",end='')
print(*o_b)

print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*nn_output)

def predictx(inp):
   a = np.dot(inp,h_w)
   c = a+h_b
   d = sigmoid(c)
   e = np.dot(d,o_w)
   f = e + o_b
   return round(sigmoid(*f[0]))

print(predictx([0,0]))
print(predictx([1,0]))
print(predictx([0,1]))
print(predictx([1,1]))

def test_coverage():
    assert predictx([0,0]) == 0
    assert predictx([0,1]) == 1 
    assert predictx([1,0]) == 1
    assert predictx([1,1]) == 0


   
