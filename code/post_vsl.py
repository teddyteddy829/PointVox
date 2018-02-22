import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


from vsl_main import*


	# def _gen_model(self, weights, biases):
 #        dec_fclayer1 = tf.nn.relu(tf.matmul(self.latent_feature , weights['dec_fc1']) + biases['dec_fc1'])
 #        dec_fclayer2 = tf.nn.relu(tf.matmul(dec_fclayer1 , weights['dec_fc2']) + biases['dec_fc2'])
 #        dec_fclayer2 = tf.reshape(dec_fclayer2, [self.batch_size, 2, 2, 2, 128])
 #        dec_conv1    = tf.nn.relu(tf.nn.conv3d_transpose(dec_fclayer2, weights['dec_conv1'],
 #                                  output_shape=[self.batch_size, 5, 5, 5, 64],
 #                                  strides=[1, 1, 1, 1, 1],padding='VALID') + biases['dec_conv1'])
 #        dec_conv2    = tf.nn.relu(tf.nn.conv3d_transpose(dec_conv1, weights['dec_conv2'],
 #                                  output_shape=[self.batch_size, 13, 13, 13, 32],
 #                                  strides=[1, 2, 2, 2, 1], padding='VALID') + biases['dec_conv2'])
 #        dec_conv3    = tf.nn.sigmoid(tf.nn.conv3d_transpose(dec_conv2, weights['dec_conv3'],
 #                                     output_shape=[self.batch_size, 30, 30, 30, 1],
 #                                     strides=[1, 2, 2, 2, 1], padding='VALID') + biases['dec_conv3'])
 #        return dec_conv3



class gen_model(object):
	def __init__(self, obj_res,batch_size):
		self.obj_res=obj_res #By default 1024
		self.batch_size=batch_size
		self.input_shape = [self.batch_size] + [self.obj_res]
        self.x = tf.placeholder(tf.float32, self.input_shape) #First node of the graph

        #Create the model, and the optimizer
        self._model_create()
        self._model_loss_optimizer()

        # start tensorflow session
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())



    def _weights_init(self):



    		self.weights_all = dict()
    		self.weights_all['W'] = {

    		'dec_conv1': tf.get_variable(name='dec_conv1', shape=[4,4,4,64, 1024],
                                          initializer=layers.xavier_initializer()),
            'dec_conv2': tf.get_variable(name='dec_conv2', shape=[6, 6, 6, 32, 64],
                                          initializer=layers.xavier_initializer()),
            'dec_conv3': tf.get_variable(name='dec_conv3', shape=[8, 8, 8, 1, 32],
                                          initializer=layers.xavier_initializer())
            }

            self.weights_all['b'] = {
            'dec_conv1': tf.Variable(name='dec_conv1', initial_value=tf.zeros(64)),
            'dec_conv2': tf.Variable(name='dec_conv2', initial_value=tf.zeros(32)),
            'dec_conv3': tf.Variable(name='dec_conv3', initial_value=tf.zeros(1)),
            }


   	def generate_model(self,weights,bias):
   		x_tranform=np.reshape(x,(self.batch_size,1,1,1,1024))
   		dec_conv1    = tf.nn.relu(tf.nn.conv3d_transpose(x_transform, weights['dec_conv1'],
                                  output_shape=[self.batch_size, 4, 4, 4, 64],
                                  strides=[1, 1, 1, 1, 1],padding='VALID') + biases['dec_conv1'])
        dec_conv2    = tf.nn.relu(tf.nn.conv3d_transpose(dec_conv1, weights['dec_conv2'],
                                  output_shape=[self.batch_size, 12, 12, 12, 32],
                                  strides=[1, 2, 2, 2, 1], padding='VALID') + biases['dec_conv2'])
        dec_conv3    = tf.nn.sigmoid(tf.nn.conv3d_transpose(dec_conv2, weights['dec_conv3'],
                                     output_shape=[self.batch_size, 30, 30, 30, 1],
                                     strides=[1, 2, 2, 2, 1], padding='VALID') + biases['dec_conv3'])
        return dec_conv3



    def _model_create(self):
        # load defined network structure
        self._weights_init()

        network_weights = self.weights_all

        self.x_rec = self.generate_model(network_weights['W'], network_weights['b'])

       	x_rec_flat=np.reshape(x_rec,(27000,1))
        
        




        # reconstruct training data from learned latent features
        

    def _model_loss_optimizer(self):

    	self.rec_loss= tf.mean_squared_loss(x_rec_flat,x_input_flat)
       	self.loss = tf.reduce_mean(self.rec_loss)


        # gradient clipping to avoid nan
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-5)
        gradients = optimizer.compute_gradients(self.loss)
 

 		def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)
        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
        self.optimizer = optimizer.apply_gradients(clipped_gradients)


    def model_fit(self, x):
        opt, cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: x})
        return cost


def main():
	data = h5py.File('dataset/ModelNet40_res30_raw.mat')

	train_all = np.transpose(data['train'])
	test_all  = np.transpose(data['test'])




	# load Gen model
	genM = gen_model(obj_res=1024, batch_size=20)

	# genM.saver.restore(genM.sess, os.path.abspath('parameters/modelnet40-2619-cost-1.1170.ckpt'))

	for epoch in range(total_epoch):
	    cost     = 0.0
	    avg_cost = 0.0
	    train_batch = int(train_all.shape[0] / batch_size)

	    index = epoch  # correct the training index, set 0 for training from scratch

	    # iterate for all batches
	    np.random.shuffle(train_all)
	    for i in range(train_batch):
	        x_train = train_all[batch_size*i:batch_size*(i+1),1:]
	        x_train = pointnet_pre_func(x_train) #Pointnet ready
	        feature_vector=pointnet.to_feature(x_train) #pointnet passed 
	        # calculate and average kl and vae loss for each batch
	        # cost[0] = np.mean(VSL.sess.run(VSL.kl_loss_all, feed_dict={VSL.x: x_train}))
	        # cost[1] = np.mean(VSL.sess.run(VSL.rec_loss, feed_dict={VSL.x: x_train}))
	        cost[0] = genM.model_fit(feature_vector)
	        avg_cost += cost / train_batch



  ##Uses the 1024X1 global feature representation and calls upon the function to generate 30X30X30 matrix

	

	






if __name__ == '__main__':
    main()
