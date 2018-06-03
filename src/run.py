'''This script is used for freezing a trained model.

It creates a model that contains only desired nodes for faster inference and remove training nodes.
Path of model to freeze is given as argument.'''

import sys, json, base64
import numpy as np
import tensorflow as tf
import cv2

file = sys.argv[-1]

if file == 'run.py':
    print ("Error loading video")
    quit()

# Define encoder function
def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

# Result of processed file
answer_key = {}

# Load model
frozen_graph = "Model/frozen_model.pb"
batch_frames = 1

# Tensor names to retrieve
input_tensor_name = "input_img:0"
training_tensor_name = "is_training:0"
output_tensor_name = "output:0"
nodes = ["input_img", "is_training", "output"]

def load_model(frozen_graph_path):
    '''Load a frozen model (adapted from http://cv-tricks.com/how-to/freeze-tensorflow-models/).
    
    Args:
        frozen_graph_path: path of the frozen model

    Returns:
        Loaded graph.
    '''
    graph = tf.get_default_graph()
    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements="",
        name="")
    return graph

# Load the frozen graph
graph = load_model(frozen_graph)

# Retrieve tensors
input_tensor = graph.get_tensor_by_name(input_tensor_name)
output_tensor = graph.get_tensor_by_name(output_tensor_name)
training_tensor = graph.get_tensor_by_name(training_tensor_name)

# Process video frame by frame
video = cv2.VideoCapture(file)
i=1

# Create a Tensorflow session
sess = tf.Session()
while(video.isOpened()):
    ret, frame = video.read()
    if ret==True:   # we have still frames to process
        # Run inference
        output = sess.run(output_tensor, feed_dict = {training_tensor: False, input_tensor: [frame]})[0]

        # Extract road and cars
        roads = (output[...,0] > 0.5).astype('uint8')
        cars = (output[...,1] > 0.5).astype('uint8')

        # Encode result
        answer_key[i] = [encode(cars), encode(roads)]
        i+=1

    else:   # end of the video
        break
        
video.release()

# Print result
print(json.dumps(answer_key))