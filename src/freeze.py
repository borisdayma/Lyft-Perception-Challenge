'''This script is used for freezing a trained model.

It creates a model that contains only desired nodes for faster inference and remove training nodes.
Path of model to freeze is given as argument.'''

import sys
import tensorflow as tf

model_path = sys.argv[-1]

if model_path == 'freeze.py':
    print ("Error loading script")
    quit()

answer_key = {}

# Name of frozen graph
frozen_graph = "frozen_model.pb"

# Tensor names to retrieve
input_tensor_name = "input_img:0"
training_tensor_name = "is_training:0"
output_tensor_name = "output:0"
nodes = ["input_img", "is_training", "output"]

# Freeze model (adapted from http://cv-tricks.com/how-to/freeze-tensorflow-models/)
def freeze_model():
    saver = tf.train.import_meta_graph(model_path + ".meta", clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    saver.restore(sess, model_path)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        nodes)
    with tf.gfile.GFile(frozen_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()

# Load frozen model (adapted from http://cv-tricks.com/how-to/freeze-tensorflow-models/)
def load_model(frozen_graph_path):
    graph = tf.get_default_graph()
    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements="",
        name="")

freeze_model()
