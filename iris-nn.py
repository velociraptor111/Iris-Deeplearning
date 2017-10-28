import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

n_nodes_hl1 = 256
n_classes = 3
x = tf.placeholder('float',[None,4])
y = tf.placeholder('float')

def get_iris_data():
    #Retrieving Data
    iris  = datasets.load_iris()
    features = iris['data'].astype(np.float32)
    target = iris['target']
    batch_size= 1
    
    num_labels = len(np.unique(target))
    
    #Convert the y labels into one-hot encoding
    all_Y = np.eye(num_labels)[target]
    
    X_train,X_test,y_train,y_test = train_test_split(features,all_Y, test_size=0.2, random_state=30)
    X_test = X_test.reshape(len(X_test),4)

    return X_train,X_test,y_train,y_test

def feed_forward_network(features):
    hidden_layer = { 'weights' : tf.Variable(tf.random_normal([4,n_nodes_hl1])),
                   'bias' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_classes]))}
    
    layer_1 = tf.add(tf.matmul(features,hidden_layer['weights']) ,hidden_layer['bias'] )
    layer_1 = tf.nn.relu(layer_1)
    
    output = tf.matmul(layer_1,output_layer['weights'])
    
    return output

def train_neural_network(x):
    
    #Get iris data
    X_train,X_test,y_train,y_test = get_iris_data()
    
    prediction = feed_forward_network(x)
    
    #Tensor to get the cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    
    #Tensor to change the weights to optimize the cost function
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(0,hm_epochs):
            #This is the lost function after each epoch
            epoch_loss = 0
            for i in range(0,int(len(X_train))):
                epoch_x = X_train[i]
                epoch_x = epoch_x.reshape(1,4)
                epoch_y = y_train[i]
                
                #_ is a convention to say that ignore this variable. The return of optimizer is nothing
                _ , c = sess.run([optimizer,cost],feed_dict = {x: epoch_x,y:epoch_y})
                
                epoch_loss += c
                
            print('Epoch', epoch,"completed out of", hm_epochs, "loss:",epoch_loss)
            
            
        #Create the tensors for evaluation!
        #We use argmax because since we used a one-hot encoding, we need
        #to find the highest probabiity
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        
        #Cast the numbers to floating point numbers and find the mean
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        
        
        print('Accuracy:', accuracy.eval({x:X_test,y:y_test}))
            

#Pass in a tensor variable for further evaluation later
train_neural_network(x)