'''This script is used for training a neural network for Semantic Segmentation.

Hyper-parameters are defined in the main function.
The script creates a neural network and train it on provided datasets.'''

import tensorflow as tf
import random
from glob import glob
import time

def decode_data(input_path, output_path):
    '''Create input and output tensors from path of labeled data.
    
    Args:
        input_path: path of camera frame
        output_path: path of labeled frame
        
    Returns:
        Set of tensors related to the "input frame" and "labeled data".        
    '''

    # read input
    image_string = tf.read_file(input_path)
    input_decoded = tf.image.decode_image(image_string)

    # read output
    image_string = tf.read_file(output_path)
    output_decoded = tf.image.decode_image(image_string)
    
    # Extract road and vehicle indices
    road = tf.cast(tf.logical_or(tf.equal(output_decoded[:,:,0], 6), tf.equal(output_decoded[:,:,0], 7)), tf.uint8)
    vehicle = tf.cast(tf.equal(output_decoded[:496,:,0], 10), tf.uint8)  # we limit to 496 to crop the hood
    vehicle = tf.concat([vehicle, tf.zeros_like(output_decoded[496:,:,0])], axis = 0)
    label = tf.stack([road, vehicle], axis = -1)

    return input_decoded, label

def augment_data(input_img, label_img, random_flip):
    '''Augment data with random operations.
    
    Args:
        input_img: Camera frame
        label_img: Labeled data
        random_flip: bool for randomly flipping vertically the image
        
    Returns:
        Set of tensors randomly augmented.
    '''

    if random_flip:
        stacked_image = tf.concat([input_img, label_img], axis=-1)
        stacked_image = tf.image.random_flip_left_right(stacked_image)
        input_img = stacked_image[:,:,:3]
        label_img = stacked_image[:,:,3:]
    return input_img, label_img


def create_datasets(random_flip, batch_size, train_path, validation_path):
    '''Create training and validation datasets.
    
    Args:
        random_flip: bool for randomly flipping vertically the image
        bach_size: size of each batch returned by dataset iterators
        train_path: root path of training data
        validation_path: root path of validation data
        
    Returns:
        Set of training and validation datasets.
    '''

    # Create training dataset
    input_files = glob(train_path + "/CameraRGB/*")
    random.shuffle(input_files)
    training_dataset = create_dataset(input_files, batch_size, random_flip)

    # Create validation dataset (no need to augment this one)
    input_files = glob(validation_path + "/CameraRGB/*")
    random.shuffle(input_files)
    validation_dataset = create_dataset(input_files, batch_size)

    return training_dataset, validation_dataset


def create_dataset(input_files, batch_size, random_flip = False):
    '''Create a dataset from a list of input files.
    
    Args:
        input_files: list of paths of input files
        batch_size: size of each batch returned by the dataset iterator
        random_flip: bool for randomly flipping vertically the image
        
    Returns:
        Tensorflow dataset.
    '''

    output_files = [f.replace('CameraRGB', 'CameraSeg') for f in input_files]
    input_files = tf.constant(input_files)
    output_files = tf.constant(output_files)
    dataset = tf.data.Dataset.from_tensor_slices((input_files, output_files))
     
    # Pre-load and process our data
    dataset = dataset.map(decode_data)

    # Shuffle the data and create batches
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(500))  # Ensure we shuffle all the files
   
    # Augment data
    dataset = dataset.map(lambda x, y: augment_data(x, y, random_flip))

    # Return data in batch
    dataset = dataset.batch(batch_size)

    return dataset


def create_mobile_unet(x, labels, is_training, learning_rate, resize_y = 256, resize_x = 256, n_layers = 3, dropout = 0.1, regularizer = False, batch_norm = False, n_classes = 2, n_filters_init = 32):
    '''Create a neural network inspired from U-net and MobileNet.

    The general architecture of U-net is used but add separable depthwise convolutions from MobileNet.
    Additional layers (batch normalization, dropout, resizing, L2 regularization) can be customized.
    Loss is based on the F-score and we use an Adam optimizer.
    
    Args:
        x: input tensor representing a set of pictures
        is_training: Placeholder bool to define whether we are in training or inference mode
        learning_rate: starting learning rate used by the Adam optimizer
        resize_y, resize_x: used for resizing of the input, can be set to None
        n_layers: number of layers in the U-net architecture
        dropout: proportion of neurons that are dropped, can be set to None
        regularizer: bool for adding a L2 loss to the total loss
        batch_norm: bool for using Batch Normalization
        n_classes: number of classes to predict
        n_filters_init: number of filters in the initial layer of the network

    Returns:
        Tensor: total loss of the network based on F-score with beta=2 for cars and beta=0.5 for road
        Tensor: regularization losses
        Tensor: training operator
        Tensor: F-score for road detection (beta = 0.5)
        Tensor: F-score for vehicle detection (beta = 2)
    '''

    regularizer = tf.contrib.layers.l2_regularizer(1e-3) if regularizer else None
    batch_norm = tf.layers.batch_normalization if batch_norm else None

    def net_conv2d(input_layer, n_filters_out):
        '''Convenience function for 2D Convolution.
        
        Args:
            input_layer: input layer of the convolution
            n_filters_out: number of filters (depth) of resulting layer
            
        Returns:
            2D convolution Layer.
        '''

        layer = tf.contrib.layers.conv2d(
            inputs = input_layer,
            num_outputs = n_filters_out,
            kernel_size = 3,
            stride = 1,
            padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=regularizer,
            activation_fn=tf.nn.elu,
            normalizer_fn=batch_norm,
            normalizer_params={"training": is_training, "fused": True},
            trainable=True)
        if dropout:
            layer = tf.contrib.layers.dropout(
                inputs = layer,
                keep_prob = 1-dropout,
                is_training = is_training)
        return layer

    def net_separable_conv2d(input_layer, n_filters_out):
        '''Convenience function for Separable Convolution.
        
        Args:
            input_layer: input layer of the convolution
            n_filters_out: number of filters (depth) of resulting layer
            
        Returns:
            Separable convolution Layer.
        '''

        layer = tf.layers.separable_conv2d(
            inputs = input_layer,
            filters = n_filters_out,
            kernel_size = 3,
            strides=(1, 1),
            padding='SAME',
            activation=tf.nn.elu,
            depthwise_initializer=tf.contrib.layers.xavier_initializer(),
            pointwise_initializer=tf.contrib.layers.xavier_initializer(),
            depthwise_regularizer=regularizer,
            pointwise_regularizer=regularizer,
            trainable=True
        )
        if batch_norm:
            layer=tf.layers.batch_normalization(layer, training = is_training, fused = True)
        if dropout:
            layer = tf.contrib.layers.dropout(
                inputs = layer,
                keep_prob = 1-dropout,
                is_training = is_training)
        return layer

    def net_up_conv(input_layer, n_filters_out):
        '''Convenience function for Up-Convolution.
        
        Args:
            input_layer: input layer of the convolution
            n_filters_out: number of filters (depth) of resulting layer
            
        Returns:
            Up-convolution Layer.
        '''
        layer = tf.layers.conv2d_transpose(
            inputs = input_layer,
            filters = n_filters_out,
            kernel_size = 2,
            strides=2,
            padding='SAME',
            activation=tf.nn.elu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=regularizer,
            trainable=True)
        return layer


    # We will need to keep track of some layers for future concatenation
    layers = []

    # Decrease size of layers and crop image
    crop_y1, crop_y2 = 170, 522
    x = tf.to_float(x)
    x_mod = x[:,crop_y1:crop_y2,:,:]
    labels = labels[:,crop_y1:crop_y2,:]

    # Normalize input
    x_mod=tf.layers.batch_normalization(x_mod, training = is_training, fused = True)

    # Resize input layer
    if resize_x:
        x_mod = tf.image.resize_images(x_mod, (resize_y, resize_x))
    
    # First layer is standard convolution + separable convolution
    n_filters = n_filters_init
    layer = net_conv2d(input_layer = x_mod, n_filters_out = n_filters)
    layer = net_separable_conv2d(input_layer = layer, n_filters_out = n_filters)

    # Down layers are made of Max-Pooling + 2 sets of separable convolutions
    for n_layer in range(1, n_layers):
        layers.append(layer)    # layer should be:
                                # [0]: 600 x 800 x (n_filters_init)
                                # [1]: 300 x 400 x (n_filters_init x 2)
                                # [2]: 150 x 200 x (n_filters_init x 4)
                                # etc (if layers > 3)
        n_filters *= 2
        layer = tf.layers.max_pooling2d(inputs=layer, pool_size=2, strides=2)
        layer = net_separable_conv2d(input_layer = layer, n_filters_out = n_filters)
        layer = net_separable_conv2d(input_layer = layer, n_filters_out = n_filters)

    # Up layers are made of Transposed convolution + 2 sets of separable convolution
    for n_layer in range(1, n_layers):
        n_filters //= 2
        layer = net_up_conv(input_layer = layer, n_filters_out = n_filters)
        layer = tf.concat([layer, layers[-n_layer]], axis = -1)
        layer = net_separable_conv2d(input_layer = layer, n_filters_out = n_filters)
        layer = net_separable_conv2d(input_layer = layer, n_filters_out = n_filters)

    # Create output classes with a depthwise convolution
    layer = tf.contrib.layers.conv2d(
            inputs = layer,
            num_outputs = n_classes,
            kernel_size = 1,
            stride = 1,
            padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=regularizer,
            activation_fn=None,
            trainable=True)

    # Use batch norm if applicable to transmit better the gradient (due to sigmoid function)
    if batch_norm:
        layer=tf.layers.batch_normalization(layer, training = is_training, fused = True)
    
    # Perform sigmoid for probability distribution with dice loss
    sigm_logits = tf.sigmoid(layer)

    # Resize to original dimensions
    if resize_x:
        y_area = crop_y2 - crop_y1
        sigm_logits = tf.image.resize_images(sigm_logits, (y_area, 800))

    # Create output tensor and prediction tensor
    output_tensor = tf.concat([tf.zeros_like(x[:,:crop_y1,:,:2]), sigm_logits, tf.zeros_like(x[:,crop_y2:,:,:2])], axis = 1, name='output')
    predict_tensor = tf.greater(output_tensor, 0.5, name="prediction")

    # F-score loss
    eps = 1e-5   # to avoid numerical issue
    beta_sq_vehicle = 2 * 2
    beta_sq_road = 0.5 * 0.5
    labels = tf.to_float(labels)
    inter_road = tf.reduce_sum(tf.multiply(labels[..., 0], sigm_logits[..., 0]), [1,2])   # Reduce image by image
    union_road = tf.reduce_sum(tf.add(beta_sq_road * labels[..., 0], sigm_logits[..., 0]), [1,2])
    inter_vehicle = tf.reduce_sum(tf.multiply(labels[..., 1], sigm_logits[..., 1]), [1,2])   # Reduce image by image
    union_vehicle = tf.reduce_sum(tf.add(beta_sq_vehicle * labels[..., 1], sigm_logits[..., 1]), [1,2])
    score_road = tf.reduce_mean((1 + beta_sq_road) * tf.divide(inter_road + eps, union_road + eps))
    score_vehicle = tf.reduce_mean((1 + beta_sq_vehicle) * tf.divide(inter_vehicle + eps, union_vehicle + eps))
    loss = 1 - (score_road + score_vehicle) / 2

    # We add regularization losses
    reg_loss = tf.losses.get_regularization_loss()
    loss += reg_loss

    # Create optimizer and training operator
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Ensure we update batch norm
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return loss, reg_loss, train_op, score_road, score_vehicle


if __name__ == '__main__':

    # Main parameters
    training_epoch_samples = 2500   # Number of samples per training epoch
                                    # This is set up so that 10% of time only is used for validation
    validation_epoch_samples = 1000 # Number of samples per validation epoch. We take our full batch.
    batch_size = 2                  # Number of samples used per batch
    n_layers = 5                    # Number of layers of network
    resize_y, resize_x = 352, 400   # Resizing parameters for input frame
    n_filters_init = 64             # Number of filters in initial layer
    learning_rate_val = 1e-3        # Adam optimizer initial learning rate
    dropout, regularizer, batch_norm = 0.1, False, True # Use of dropout, L2 regularization and Batch Normalization
    random_flip = False             # Data augmentation through random vertical flipping
    model_name = "../Model/model.ckpt"  # Path of training model
    reload_training = False             # Reload from previous model
    training_data_path = "../big_data/Train"        # Path of training data
    validation_data_path = "../big_data/Validation" # Path of validation data

    # Load datasets
    learning_rate = tf.placeholder(tf.float64)
    dataset_train, dataset_valid = create_datasets(random_flip, batch_size, training_data_path, validation_data_path)
    
    # Create feedable iterator
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, dataset_train.output_types, dataset_train.output_shapes)
    input_img, labels = iterator.get_next()

    # Create iterators for training and validation datasets
    iterator_train, iterator_valid = dataset_train.make_one_shot_iterator(), dataset_valid.make_one_shot_iterator()

    # Prepare input data for network
    input_img = tf.reshape(input_img, (-1, 600, 800, 3), name='input_img')
    labels = tf.reshape(labels, (-1, 600, 800, 2), name='processed_label_img')

    # Create network
    is_training = tf.placeholder(tf.bool, name="is_training")
    loss, reg_loss, train_op, score_road, score_vehicle = create_mobile_unet(x=input_img, labels=labels, is_training=is_training, resize_y=resize_y, resize_x = resize_x,
                                                               n_filters_init = n_filters_init, n_layers=n_layers,
                                                               dropout = dropout, regularizer = regularizer, batch_norm = batch_norm, learning_rate=learning_rate)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Write total number of network parameters for information
    total_parameters = 0
    for variable in tf.all_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total network parameters: ", total_parameters)

    with tf.Session() as sess:
        # Load network if already existing
        if reload_training:
            saver.restore(sess, model_name)
        else:
            # Initialize data
            sess.run(tf.global_variables_initializer())

        # Initialize iterator handles
        training_handle = sess.run(iterator_train.string_handle())
        validation_handle = sess.run(iterator_valid.string_handle())

        # Define number of epochs for training and validation data
        epochs_train = int(training_epoch_samples // batch_size)
        epochs_valid = int(validation_epoch_samples // batch_size)
        start_time = time.time()

        # Keep track of best validation loss
        best_validation_loss = 1e5
        epoch_iteration = 0

        while True:

            epoch_iteration += 1
            print("Epoch {}".format(epoch_iteration))
            
            # Perform training on full dataset
            total_loss, total_reg_loss, total_score_road, total_score_vehicle = 0, 0, 0, 0
            epochs = epochs_train
            # Run the network until we reach our target number of samples for one epoch
            # based on batch_size and training_epoch_samples
            for _ in range(epochs):
                loss_value, reg_loss_value,  _, score_road_value, score_vehicle_value = sess.run([loss, reg_loss, train_op, score_road, score_vehicle],
                    feed_dict = {is_training: True, learning_rate: learning_rate_val, handle:training_handle})
                total_loss += loss_value / epochs
                total_reg_loss += reg_loss_value / epochs
                total_score_road += score_road_value / epochs
                total_score_vehicle += score_vehicle_value / epochs
            print("Training loss: {}, Reg. loss: {}, Score Road: {}, Score Car: {}, Time: {}".format(total_loss, total_reg_loss, total_score_road, total_score_vehicle, time.time() - start_time))
            
            # Check performance on validation data
            epochs = epochs_valid
            total_loss, total_reg_loss, total_score_road, total_score_vehicle = 0, 0, 0, 0
            for _ in range(epochs):
                loss_value, reg_loss_value, score_road_value, score_vehicle_value = sess.run([loss, reg_loss, score_road, score_vehicle],
                    feed_dict = {is_training: False, handle:validation_handle})
                total_loss += loss_value / epochs
                total_reg_loss += reg_loss_value / epochs
                total_score_road += score_road_value / epochs
                total_score_vehicle += score_vehicle_value / epochs
            print("Validation loss: {}, Reg. loss: {}, Score Road: {}, Score Car: {}, Time: {}".format(total_loss, total_reg_loss, total_score_road, total_score_vehicle, time.time() - start_time))

            # Save best model
            if total_loss < best_validation_loss:
                best_validation_loss = total_loss
                save_path = saver.save(sess, model_name)
                print("Model saved in path: ", save_path)
            else:
                # We still keep intermediate models
                save_path = saver.save(sess, model_name + "intermediate")