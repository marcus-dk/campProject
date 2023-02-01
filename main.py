"""
our implementation of an image recognition neural network from scratch
made by marcus and laurits, no tutorials etc. used - only documentation and such

"""
import tensorflow as tf # only used to download the data
import numpy as np
import matplotlib.pyplot as plt


# loading of dataset from tensorflow
number_mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = number_mnist.load_data()

# preprocessing of images 
train_images = train_images / 255.0
test_images = test_images / 255.0


# activation function definitions 
# ---------------------------------------------------

# relu activation function
relu = lambda x: np.maximum(0, x)

# derivative relu function
deriv_relu = lambda x: 1 if x > 0 else 0

# softmax activation function
softmax = lambda x: np.exp(x)/np.sum(np.exp(x), axis=0)

# ---------------------------------------------------

def cross_entropy_loss(predictions, labels):
    """Calculates the cross-entropy loss between predicted output and true labels."""
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    loss = -np.sum(labels * np.log(predictions))
    return loss

def onehot_label(label):
    ohlabel = np.array([0 for i in range(0,10)])
    ohlabel[label] = 1
    return ohlabel

# for development with laurits on 2 computers
# generates start weights for nodes, multiplied by 2 and subtract one to get weights from -1->1 instead of only 0->1 
#weight_gen = lambda num_inodes, index: 2 * np.random.random((num_inodes, 1)) - 1

def weight_gen(num_inodes, index):
    np.random.seed(index)
    return 2 * np.random.random((num_inodes, 1)) - 1
    #return [2 * np.random.random((num_inodes, 1))-1 for i in range()]
# flattens 2d array of image, this is done to avoid having to have 700+ input nodes, and instead just 10.
flatten = lambda image_array: np.ndarray.flatten(image_array)

dotProduct_input_and_weight = lambda inputdata, weights: np.dot(inputdata,weights)

# loss function --- (outcome - weightoutcome)^2

def iL2hL(inputNodes, image, hLNodes):
    outgoingNodes = np.array([])
    weights = np.array([weight_gen(inputNodes,i) for i in range(0,hLNodes)])
 
    for i in range(0,hLNodes):
            
        outgoingNodes = np.append(outgoingNodes, relu(dotProduct_input_and_weight(image,weights[i]))) 
        

    return outgoingNodes, weights

def hL2oL(hlNodes, hiddenLayer, oLNodes):
    outgoingNodes = np.array([])
    weights = np.array([weight_gen(hlNodes,i) for i in range(0,oLNodes)])
    
    for i in range(0, oLNodes):
            
        outgoingNodes = np.append(outgoingNodes, dotProduct_input_and_weight(hiddenLayer,weights[i]))

    return outgoingNodes, weights


# actual neural network
def nn(train_images, train_labels):
    
    inputNodes = 784 
    outputNodes = 10
    hLNodes = 100  
    
    flattened_train_images = np.array([flatten(toflatten_image) for toflatten_image in train_images])
    
    hl = iL2hL(inputNodes, flattened_train_images[0], hLNodes)
    hlweights = hl[1]
    hlvalues = hl[0]
   
    ol = hL2oL(hLNodes, hlvalues, outputNodes)
    olweights = ol[1]
    olvalues = ol[0]
    softmaxolvalues = softmax(ol[0])
    
    one_hotlabel = onehot_label(train_labels[0])

    loss = cross_entropy_loss(softmaxolvalues, one_hotlabel)
    dreLUloss = deriv_relu(loss)
    footstep = 0.00001  

    print(dreLUloss)

    #print(softmaxolvalues)
    #print(train_labels[0])
    #print(olweights)

# remember to change counter seed to random seed



# how do i backpropagate my mnist neural network based off of my weights and the cross entropy loss

nn(train_images,train_labels)
