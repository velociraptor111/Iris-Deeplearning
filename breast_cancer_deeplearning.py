import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

num_features = 9
n_nodes_hl1 = 5 #Mean of the input and output... 9+2/2 = 5
n_classes = 2 #Benign or Malignant
x = tf.placeholder('float',[None,num_features])
y = tf.placeholder('float')

def get_data():
    breast_cancer_data = pd.read_csv('breast-cancer-wisconsin.data.txt')
    breast_cancer_data.replace('?',-99999 , inplace=True)
    breast_cancer_data['bare_nuclei'] = breast_cancer_data['bare_nuclei'] .astype('float64')
    breast_cancer_data.drop(['id'], axis=1,inplace=True)
    targets = np.array(breast_cancer_data['class'])
    num_labels = len(np.unique(targets))
    
    targets[targets == 2] = 0
    targets[targets == 4] = 1
    #Convert the y labels into one-hot encoding
    all_Y = np.eye(num_labels)[targets]
    features = breast_cancer_data.drop('class',axis=1)
    features = features.as_matrix()
    
    X_train,X_test,y_train,y_test = train_test_split(features,all_Y, test_size=0.2, random_state=66)

    
    return X_train,X_test,y_train,y_test
    

def feed_forward_network(features):
    hidden_layer = { 'weights' : tf.Variable(tf.random_normal([9,n_nodes_hl1])),
                   'bias' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = { 'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_classes]))}
    
    layer_1 = tf.add(tf.matmul(features,hidden_layer['weights']) ,hidden_layer['bias'] )
    layer_1 = tf.nn.relu(layer_1)
    
    output = tf.matmul(layer_1,output_layer['weights'])
    
    return output

def train_neural_network(x):
    
    #Get iris data
    X_train,X_test,y_train,y_test = get_data()
    
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
                epoch_x = epoch_x.reshape(1,9)
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
 

train_neural_network(x)