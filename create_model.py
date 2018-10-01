import tflearn 
import tensorflow as tf

#AND GATE

X = [[0,0], [0,1], [1,1], [1, 0]]
Y = [[0], [0], [1], [0]]


#INITIALIZING WEIGHTS TO RANDOM VALUES THROUGHOUT THE ENTIRE NETWORK
weights = tflearn.initializations.uniform(minval = -1, maxval = 1)

#EACH LAYER OF THE NEURAL NETWORK MUST BE BUILT SEPARATELY, EXCEPT THE 
#HIDDEN LAYER, FOR WHICH YOU NEED TO SPECIFIY HOW MANY NEURONS MUST BE 
#USED IN EACH LAYER. 

#INPUT LAYER 
net = tflearn.input_data(shape = [None, 2], name = 'input')

#HIDDEN LAYERS 
net = tflearn.fully_connected(net, 4, activation = 'sigmoid', weights_init = weights)
net = tflearn.fully_connected(net, 3, activation = 'sigmoid', weights_init = weights)

#OUTPUT LAYER
net = tflearn.fully_connected(net, 1, activation = 'sigmoid', weights_init = weights, name = 'output')

#REGRESSION AND LEARNING LAYER
net = tflearn.regression(net, learning_rate = 2, optimizer = 'sgd', loss = 'mean_square')

#STATING THAT THIS IS A DEEP NEURAL NETWORK
model = tflearn.DNN(net)

#FIT THE TRAINING DATA INTO THE MODEL FIRST BEFORE PREDICTING
model.fit(X, Y, 5000)

#PREDICTING 0 AND 1 
print("0 AND 0 = %f" % model.predict([[0,0]]).item(0))
print("0 AND 1 = %f" % model.predict([[0,1]]).item(0))
print("1 AND 0 = %f" % model.predict([[1,0]]).item(0))
print("1 AND 1 = %f" % model.predict([[1,1]]).item(0))


'''
#BEFORE SAVING THE MODEL, TFMOBILE MANDATES THE REMOVAL OF TRAINING-RELATED OPERATOINS 
with net.graph.as_default():
	del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
	
#SAVE MODEL
model.save('and.tflearn')
'''





