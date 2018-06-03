'''This script is used for testing a trained neural network on Semantic Segmentation.

It can be used for inference on sample videos or displaying predictions on datasets.
Any video to process needs to be given as argument to the script.'''

from moviepy.editor import VideoFileClip, ImageSequenceClip
from PIL import Image
import numpy as np
import argparse, sys, cv2, os
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import train

input_file = sys.argv[-1]

# Main variables
model_path = "..\Model\model_202.ckpt"          # Path of model to load
output_file = "result.mp4"                      # Path of output video
training_data_path = "../big_data/Train"        # Path of training dataset
validation_data_path = "../big_data/Validation" # Path of validation dataset

# Tensor names to retrieve
input_tensor_name = "input_img:0"
predict_tensor_name = "prediction:0"
training_tensor_name = "is_training:0"
processed_label_tensor_name = "processed_label_img:0"
label_tensor_name = "IteratorGetNext:1"
learning_rate_tensor_name = "Placeholder:0"
iterator_handle_tensor_name = "Placeholder_1:0"
output_tensor_name = "output:0"

# Restore model
saver = tf.train.import_meta_graph(model_path + ".meta")
sess = tf.Session()
saver.restore(sess, model_path)
input_tensor = tf.get_default_graph().get_tensor_by_name(input_tensor_name)
predict_tensor = tf.get_default_graph().get_tensor_by_name(predict_tensor_name)
training_tensor = tf.get_default_graph().get_tensor_by_name(training_tensor_name)
label_tensor = tf.get_default_graph().get_tensor_by_name(label_tensor_name)
processed_label_tensor = tf.get_default_graph().get_tensor_by_name(processed_label_tensor_name)
learning_rate_tensor = tf.get_default_graph().get_tensor_by_name(learning_rate_tensor_name)
iterator_handle_tensor = tf.get_default_graph().get_tensor_by_name(iterator_handle_tensor_name)
output_tensor = tf.get_default_graph().get_tensor_by_name(output_tensor_name)

# Create dataset iterators
dataset_train, dataset_valid = train.create_datasets(True, 1, training_data_path, validation_data_path)
iterator_train, iterator_valid = dataset_train.make_one_shot_iterator(), dataset_valid.make_one_shot_iterator()

# Get iterator handles
training_handle = sess.run(iterator_train.string_handle())
validation_handle = sess.run(iterator_valid.string_handle())

# Load and create iterator for test dataset
input_files = glob("../big_data/Test/*")
output_files = [f.replace('CameraRGB', 'CameraSeg') for f in input_files]
input_files = tf.constant(input_files)
output_files = tf.constant(output_files)
dataset = tf.data.Dataset.from_tensor_slices((input_files, output_files))
dataset = dataset.map(train.decode_data)
dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(500))
iterator_test = dataset.make_one_shot_iterator()
test_handle = sess.run(iterator_test.string_handle())

def process_frame(rgb_frame = None, debug_img = False):
    '''Process image samples and perform inference.
    
    Args:
        rgb_frame: input image to feed to neural network, None if we get data from the dataset
        debug_img: bool for viewing predictions on images from dataset
                   if None, it processes the frame for generating video frame
    '''   

    if debug_img:       # We will only view inference on samples from dataset

        while True:
            # Run the model for inference         
            input_frame, predict, processed_label, output = sess.run(
                [input_tensor, predict_tensor, processed_label_tensor, output_tensor], feed_dict = {training_tensor: False, iterator_handle_tensor:training_handle})        
            input_frame = input_frame[0]
            processed_label = processed_label[0]
            processed_label = processed_label[...,0] + 2 * processed_label[...,1]
            predict = predict[0]
            output = output[0]
            output_road = (output[...,0] * 255).astype('uint8')
            output_cars = (output[...,1] * 255).astype('uint8')

            # Display inference and ground truth
            f = plt.figure()
            f.add_subplot(2, 2, 1)
            plt.title('Camera Frame')
            plt.axis('off')
            plt.imshow(input_frame)
            f.add_subplot(2, 2, 2)
            plt.title('Ground Truth Labels')
            plt.axis('off')
            plt.imshow(processed_label)
            f.add_subplot(2, 2, 3)
            plt.title('Road Prediction')
            plt.axis('off')
            overlay = np.zeros_like(input_frame)
            overlay[:,:,1] = output_road
            plt.imshow(overlay)
            f.add_subplot(2, 2, 4)
            plt.title('Car Prediction')
            plt.axis('off')
            overlay = np.zeros_like(input_frame)
            overlay[:,:,0] = output_cars
            plt.imshow(overlay)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.show()

    else:       # We create a video frame with overlays

        # Add a dimension for inference to neural network
        rgb_frame = np.expand_dims(rgb_frame, 0)

        # Run the model for inference
        predict = sess.run(predict_tensor, feed_dict = {training_tensor: False, input_tensor: rgb_frame})
        rgb_frame = rgb_frame[0]
        roads = (predict[...,0]).astype('uint8') * 255
        cars = (predict[...,1]).astype('uint8') * 255
        
        # Create overlays with inference of cars and road
        overlay = np.zeros_like(rgb_frame)
        overlay[:,:,0] = cars
        overlay[:,:,1] = roads
        
        # Add overlay on input frame
        final_frame = cv2.addWeighted(rgb_frame, 1, overlay, 0.3, 0, rgb_frame)
        return final_frame

# Define pathname to save the output video
debug_img = True        # True: presents prediction from dataset, False: processes a video
if debug_img:
    process_frame(None, debug_img = True)
else:
    clip1 = VideoFileClip(input_file)
    clip = clip1.fl_image(process_frame)
    clip.write_videofile(output_file, audio=False)
