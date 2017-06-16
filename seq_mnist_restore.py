import numpy as np
import tensorflow as tf
import sys


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

total_epochs = 2
mini_batch_size = 50
train_size = mnist.train.num_examples
total_mini_batch_number = int(np.ceil(train_size/mini_batch_size))

time_steps = 28
hidden_layer_shape = 128
input_shape = 28
output_shape = 10

class Vanilla_RNN_cell(object):
    """ 
    Vanilla_RNN_cell object
    """
    
    def __init__(self, input_shape, hidden_layer_shape, output_shape):

        # Initialization of given values
        self.input_shape = input_shape
        self.hidden_layer_shape = hidden_layer_shape
        self.output_shape = output_shape

        # Weights matrices and bias

        self.Wx = tf.Variable(tf.truncated_normal([input_shape, hidden_layer_shape], mean = 0, stddev = 0.01), name = 'Wx')
        self.Wh = tf.Variable(tf.truncated_normal([hidden_layer_shape, hidden_layer_shape], mean = 0, stddev = 0.01), name = 'Wh')
        self.Wo = tf.Variable(tf.truncated_normal([hidden_layer_shape, output_shape], mean = 0, stddev = 0.01), name = 'Wo')
    
        self.bias_hidden = tf.Variable(tf.zeros([hidden_layer_shape]), name = 'bias_hidden')
        self.bias_out = tf.Variable(tf.zeros([output_shape]), name = 'bias_out')

        tf.add_to_collection('vars', self.Wx)
        tf.add_to_collection('vars', self.Wh)
        tf.add_to_collection('vars', self.Wo)
        tf.add_to_collection('vars', self.bias_hidden)
        tf.add_to_collection('vars', self.bias_out)


    def vanilla_rnn(self, previous_hidden_state, current_input):
        """
        Vanilla rnn "cell"
        Takes previous hidden state and current input and returns current hidden state 
        """

        current_hidden_state = tf.tanh(tf.matmul(previous_hidden_state, self.Wh) + tf.matmul(current_input, self.Wx) + self.bias_hidden)
        return current_hidden_state
        
    def get_states(self, inputs, initial_state):
        """
        Iterates through time to get all hidden states
        """
        
        current_hidden_state = initial_state
        all_hidden_states = []

        for t in range(time_steps):
            new_hidden_state = self.vanilla_rnn(current_hidden_state, inputs[t])
            all_hidden_states.append(new_hidden_state)
            current_hidden_state = new_hidden_state

        #all_hidden_states = tf.scan(self.vanilla_rnn, processed, initializer = self.hidden_0, name = 'states')

        return current_hidden_state

    def get_output(self, inputs, initial_state):
        """
        Takes hidden state and returns output
        """

        last_hidden_state = self.get_states(inputs, initial_state)
        output = tf.nn.relu(tf.matmul(last_hidden_state, self.Wo) + self.bias_out)

        return output

targets = tf.placeholder(tf.float32, shape = [None, output_shape], name = 'targets')        

input_placeholder = tf.placeholder(tf.float32, shape = [None, time_steps, input_shape], name = 'inputs')

initial_state = tf.placeholder(tf.float32, shape = [None, hidden_layer_shape], name = 'initial_state')


inputs = tf.unstack(input_placeholder, time_steps, 1)

rnn = Vanilla_RNN_cell(input_shape, hidden_layer_shape, output_shape)

last_output = rnn.get_output(inputs, initial_state)

train_step = tf.train.AdamOptimizer().minimize(tf.nn.softmax_cross_entropy_with_logits(last_output, targets))

correct_predictions = tf.equal(tf.argmax(targets, 1), tf.argmax(tf.nn.softmax(last_output), 1))

acc = (tf.reduce_mean(tf.cast(correct_predictions, tf.float32)))*100


sess = tf.Session()
initializer = tf.global_variables_initializer()
sess.run(initializer)
restore_saver = tf.train.Saver()
restore_saver.restore(sess, ('./saved_model'))


# Training loop
for epc in range(total_epochs):
    for i in range(total_mini_batch_number):


        train_mini_batch = mnist.train.next_batch(mini_batch_size)
        input_mini_batch = train_mini_batch[0].reshape([mini_batch_size, time_steps, input_shape])
        targets_mini_batch = train_mini_batch[1]
        initial_state_mini_batch = np.zeros([mini_batch_size, hidden_layer_shape])

        test_mini_batch = mnist.test.next_batch(mini_batch_size)
        test_input_mini_batch = test_mini_batch[0].reshape([mini_batch_size, time_steps, input_shape])
        test_targets_mini_batch = test_mini_batch[1]

        sess.run(train_step,feed_dict={input_placeholder: input_mini_batch, targets: targets_mini_batch, initial_state: initial_state_mini_batch})

                
    train_acc = str(sess.run(acc, feed_dict={input_placeholder: input_mini_batch, targets: targets_mini_batch, initial_state: initial_state_mini_batch}))
    test_acc = str(sess.run(acc, feed_dict={input_placeholder:test_input_mini_batch, targets: test_targets_mini_batch, initial_state: initial_state_mini_batch}))
    

    print("\r Epoch: %s Train Accuracy: %s Test Accuracy: %s" %(epc, train_acc, test_acc))