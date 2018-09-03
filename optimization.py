from __future__ import print_function
import matplotlib.pyplot as plt
import math
from io import BytesIO
import numpy as np
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import scipy
import imageio
from time import gmtime, strftime, localtime
import time
from functools import reduce, partial
import tensorflow as tf
import keras.preprocessing.image as kpimage

import vgg19

import os
import scipy.misc
import scipy.io


from os import listdir
from os.path import isfile, join

from dgn import *

BATCH_SIZE=1

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

# uncomment below for reproducability
# np.random.seed(1)

def normalize(img, out_range=(0.,1.), in_range=None):
	if not in_range:
		min_val = np.min(img)
		max_val = np.max(img)
	else:
		min_val = in_range[0]
		max_val = in_range[1]
	result = np.copy(img)
	result[result > max_val] = max_val
	result[result < min_val] = min_val
	result = (result - min_val) / (max_val - min_val) * (out_range[1] - out_range[0]) + out_range[0]
	return result



init_shape = (1,224,224,3)



vgg = vgg19.Vgg19()




#vgg_layers = (vgg.pool1, vgg.pool2, vgg.pool3, vgg.pool4, vgg.pool5, vgg.fc6, vgg.fc7)



iteration=1





def reconstruct_from_feature_no_prior(pre_match_vals,out_name,iters_dir):
	
	vgg.build()
	vgg_layers = (vgg.conv1_1,vgg.conv1_2,vgg.conv2_1,vgg.conv2_2,vgg.conv3_1,vgg.conv3_2,vgg.conv3_3,vgg.conv3_4,vgg.conv4_1,vgg.conv4_2,vgg.conv4_3,vgg.conv4_4,
		vgg.conv5_1,vgg.conv5_2,vgg.conv5_3,vgg.conv5_4, vgg.fc6, vgg.fc7,vgg.fc8) 

	upper_bound=np.loadtxt('fc6.txt',delimiter=' ',usecols=np.arange(0,4096),unpack=True)
	upper_bound=upper_bound.reshape((1,4096)).astype('float32')
	lower_bound=np.zeros((1,4096)).astype('float32')
	



	W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15, W16, W17, W18, W19=pre_match_vals





	step=2.0
	verbose=True
	iters = 500
	decay=(2.0-(1e-10))/500.0
	m=0.9




	print('SETTING UP AT: {}'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
	with tf.device('/cpu:0'), tf.Session() as sess:
		#print("Restoring Generator")
		#vgg.generator_saver.restore(sess, get_pretrain_generator(net='caffenet', load_type='tf'))
		

		benchmark1 = tf.placeholder(tf.float32, shape=(None))
		benchmark2 = tf.placeholder(tf.float32, shape=(None))
		benchmark3 = tf.placeholder(tf.float32, shape=(None))
		benchmark4 = tf.placeholder(tf.float32, shape=(None))
		benchmark5 = tf.placeholder(tf.float32, shape=(None))
		benchmark6 = tf.placeholder(tf.float32, shape=(None))
		benchmark7 = tf.placeholder(tf.float32, shape=(None))
		benchmark8 = tf.placeholder(tf.float32, shape=(None))
		benchmark9 = tf.placeholder(tf.float32, shape=(None))
		benchmark10 = tf.placeholder(tf.float32, shape=(None))
		benchmark11 = tf.placeholder(tf.float32, shape=(None))
		benchmark12 = tf.placeholder(tf.float32, shape=(None))
		benchmark13 = tf.placeholder(tf.float32, shape=(None))
		benchmark14 = tf.placeholder(tf.float32, shape=(None))
		benchmark15 = tf.placeholder(tf.float32, shape=(None))
		benchmark16 = tf.placeholder(tf.float32, shape=(None))
		benchmark17 = tf.placeholder(tf.float32, shape=(None))
		benchmark18 = tf.placeholder(tf.float32, shape=(None))
		benchmark19 = tf.placeholder(tf.float32, shape=(None))
		


		benchmarks=[benchmark1,benchmark2,benchmark3,benchmark4,benchmark5,benchmark6,benchmark7,benchmark8,
		benchmark9,benchmark10,benchmark11,benchmark12,benchmark13,benchmark14,benchmark15,benchmark16,benchmark17,
		benchmark18,benchmark19
		]

		feed={
		benchmark1:W1,benchmark2:W2,benchmark3:W3,benchmark4:W4,benchmark5:W5,benchmark6:W6,benchmark7:W7,benchmark8:W8,benchmark9:W9,
		benchmark10:W10,benchmark11:W11,benchmark12:W12,benchmark13:W13,benchmark14:W14,benchmark15:W15,benchmark16:W16,benchmark17:W17,benchmark18:W18,
		benchmark19:W19
		}



		inter_maxes = []


		for j, curr_layer in enumerate(vgg_layers):

			benchmark_layer=benchmarks[j]

			n=tf.square(tf.norm(benchmark_layer))

			weight_hi = tf.reduce_sum(tf.square(tf.subtract(curr_layer,benchmark_layer)))
			inter_maxes.append(weight_hi/n)
			
		weight_max = 0.5*tf.add_n(inter_maxes)
		optimizer=tf.contrib.opt.ScipyOptimizerInterface(weight_max,method='L-BFGS-B',var_list=[vgg.input_var],options={'maxiter':200})
		then = time.time()
		
		
		VGG_MEAN = [103.939, 116.779, 123.68]
		
		input_pre=scipy.misc.imresize(np.load('ilsvrc_2012_mean.npy'),(224,224),interp='bicubic').reshape(init_shape)


		


		sess.run(tf.global_variables_initializer(),feed_dict={vgg.input:input_pre})

		
		def loss_cb(loss,input_pre):
			global iteration
			print('ITERATION: {}/{}, FINISHED AT: {}, LOSS = {}'.format(iteration,200,strftime("%Y-%m-%d %H:%M:%S", localtime()),loss))
			img_out=input_pre[0]
			#print(img_out.shape)
			print("Saving to "+iters_dir+"iter_"+str(iteration)+".png")
			scipy.misc.imsave(iters_dir+"iter_"+str(iteration)+".png",normalize(img_out))
			iteration+=1
			
		

			


		print('STARTING AT: {}'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
		

		optimizer.minimize(session=sess,feed_dict=feed,fetches=[weight_max,vgg.input_var],loss_callback=loss_cb)




		img_out=sess.run(vgg.input_var,feed_dict={})[0]
		print(img_out.shape)
		scipy.misc.imsave(out_name,normalize(img_out))
		#np.save("losses.npy",losses)
		print('FINISHED.')


def reconstruct_from_feature_prior(pre_match_vals,out_name,iters_dir): 

	upper_bound=np.loadtxt('fc7.txt',delimiter=' ',usecols=np.arange(0,4096),unpack=True)
	upper_bound=upper_bound.reshape((1,4096)).astype('float32')
	lower_bound=np.zeros((1,4096)).astype('float32')
	
	vgg.build_with_image_prior()

	vgg_layers = (vgg.conv1_1,vgg.conv1_2,vgg.conv2_1,vgg.conv2_2,vgg.conv3_1,vgg.conv3_2,vgg.conv3_3,vgg.conv3_4,vgg.conv4_1,vgg.conv4_2,vgg.conv4_3,vgg.conv4_4,
		vgg.conv5_1,vgg.conv5_2,vgg.conv5_3,vgg.conv5_4, vgg.fc6, vgg.fc7,vgg.fc8)


	W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15, W16, W17, W18, W19=pre_match_vals





	step=2.0
	verbose=True
	iters = 200
	decay=(2.0-(1e-10))/200.0
	m=0.9




	print('SETTING UP AT: {}'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
	with tf.device('/cpu:0'), tf.Session() as sess:
		print("Restoring Generator")
		vgg.generator_saver.restore(sess, get_pretrain_generator(net='caffenet', load_type='tf'))
		

		benchmark1 = tf.placeholder(tf.float32, shape=(None))
		benchmark2 = tf.placeholder(tf.float32, shape=(None))
		benchmark3 = tf.placeholder(tf.float32, shape=(None))
		benchmark4 = tf.placeholder(tf.float32, shape=(None))
		benchmark5 = tf.placeholder(tf.float32, shape=(None))
		benchmark6 = tf.placeholder(tf.float32, shape=(None))
		benchmark7 = tf.placeholder(tf.float32, shape=(None))
		benchmark8 = tf.placeholder(tf.float32, shape=(None))
		benchmark9 = tf.placeholder(tf.float32, shape=(None))
		benchmark10 = tf.placeholder(tf.float32, shape=(None))
		benchmark11 = tf.placeholder(tf.float32, shape=(None))
		benchmark12 = tf.placeholder(tf.float32, shape=(None))
		benchmark13 = tf.placeholder(tf.float32, shape=(None))
		benchmark14 = tf.placeholder(tf.float32, shape=(None))
		benchmark15 = tf.placeholder(tf.float32, shape=(None))
		benchmark16 = tf.placeholder(tf.float32, shape=(None))
		benchmark17 = tf.placeholder(tf.float32, shape=(None))
		benchmark18 = tf.placeholder(tf.float32, shape=(None))
		benchmark19 = tf.placeholder(tf.float32, shape=(None))
		


		benchmarks=[benchmark1,benchmark2,benchmark3,benchmark4,benchmark5,benchmark6,benchmark7,benchmark8,
		benchmark9,benchmark10,benchmark11,benchmark12,benchmark13,benchmark14,benchmark15,benchmark16,benchmark17,
		benchmark18,benchmark19
		]

		feed={
		benchmark1:W1,benchmark2:W2,benchmark3:W3,benchmark4:W4,benchmark5:W5,benchmark6:W6,benchmark7:W7,benchmark8:W8,benchmark9:W9,
		benchmark10:W10,benchmark11:W11,benchmark12:W12,benchmark13:W13,benchmark14:W14,benchmark15:W15,benchmark16:W16,benchmark17:W17,benchmark18:W18,
		benchmark19:W19
		}



		inter_maxes = []


		for j, curr_layer in enumerate(vgg_layers):

			benchmark_layer=benchmarks[j]

			#n=tf.square(tf.norm(benchmark_layer))
			n=tf.reduce_sum(tf.square(benchmark_layer))

			weight_hi = tf.reduce_sum(tf.square(tf.subtract(curr_layer,benchmark_layer)))
			inter_maxes.append(weight_hi/n)
			
		weight_max = 0.5*tf.add_n(inter_maxes)

		
		
		print(vgg.generator_vars_list)
		for var in vgg.generator_vars_list:
			tf.stop_gradient([var])

		weight_grad=tf.gradients(weight_max,[vgg.vector])[0]




		then = time.time()
		
		

		input_pre=np.zeros((1,4096)).astype(np.float32)
		#sess.run(tf.global_variables_initializer(),feed_dict={vgg.vector:input_pre})

		print('STARTING AT: {}'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))

		
		for i in range(iters):
			inputvec=interim.copy() if i!=0 else input_pre.copy()
			feed[vgg.vector]=inputvec

			g, loss= sess.run([weight_grad, weight_max], feed_dict=feed)
			print('ITERATION: {}/{}, FINISHED AT: {}, LR={}, LOSS = {}'.format(i,iters,strftime("%Y-%m-%d %H:%M:%S", localtime()),step,loss))
			
			#print(inputvec)
			#Gradient Descent with Momentum
			if i==0:
				grad=-step*g
			else:	
				grad=m*grad-step*g
			input_pre=input_pre+grad
			step=step-decay

			#G=m*G+g**2
			#step=1.0/((1/2.0)+(G**0.5))
			#grad=m*grad-step*g
			#input_pre=input_pre+grad
			

			                                                        
			
			


			
			img_out=sess.run(vgg.input,feed_dict={vgg.vector:input_pre})[0]
			img_out=img_out[:,:,::-1]
			print(np.min(img_out),np.max(img_out))
			print("Saving to "+iters_dir+"iter_"+str(i)+".png")
			scipy.misc.imsave(iters_dir+"iter_"+str(i)+".png",normalize(img_out))

			input_pre=np.maximum(input_pre,lower_bound)
			input_pre=np.minimum(input_pre,upper_bound)
			interim=input_pre.copy()





		img_out=sess.run(vgg.input,feed_dict={vgg.vector:input_pre})[0]
		img_out=img_out[:,:,::-1]
		print(img_out.shape)
		#scipy.misc.imsave(out_name,img_out)
		scipy.misc.imsave(out_name,normalize(img_out,in_range=(0,255)))
		#np.save("losses.npy",losses)
		print('FINISHED.')





"""input_dir="/gpfs/milgram/project/chun/sk2436/reconstruction/test/"
names=[f[:-4] for f in listdir(input_dir) if isfile(join(input_dir,f))]
fnames=[input_dir+names[i]+".png" for i in range(len(names))]"""


#reconstruct_from_image(fnames[0],"test_recons/"+names[0]+".png")

from load import *
import pickle

#all_features=pickle.load(open("/gpfs/milgram/project/chun/sk2436/reconstruction/cnn_features2/full_recon_features.pickle","rb"))

#all_features=pickle.load(open("/gpfs/milgram/project/chun/sk2436/reconstruction/cnn_features2/vgg19_testv2.pickle","rb"))
#input_dir="/gpfs/milgram/project/chun/sk2436/reconstruction/test/"
#names_set=[f[:-4] for f in listdir(input_dir) if isfile(join(input_dir,f))]


all_features=pickle.load(open("/gpfs/milgram/project/chun/sk2436/reconstruction/cnn_features2/full_recon_featuresv3.pickle","rb"))

names,_=load_imagenet_BOLD_features('PPA',1,train=False)

names_set=sorted(list(set(names)))


print(names_set)

import sys

i=int(sys.argv[1])


name="test_recons3/model_"+names_set[i]+".png"
print(name)
iters_dir='/gpfs/milgram/project/chun/sk2436/reconstruction/iters'+str(i)+'/'
feats=all_features[i]

reconstruct_from_feature_prior(feats,name,iters_dir)


