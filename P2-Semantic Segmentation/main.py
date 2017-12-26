import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU 
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load saved models and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get each operation
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name) 
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    # return as tuple
    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

tests.test_load_vgg(load_vgg, tf)
# tests passed

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """    
    # Encoding - transfer learning from VGG
    # Classification block - converted from fully connected to fully convolutional
    x = tf.layers.conv2d(vgg_layer7_out , 4096, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu)
    x = tf.layers.conv2d(x              , 1000, kernel_size=(1,1),  strides=(1,1), activation=tf.nn.relu)


    ### Decoding - upsampling
    # 2x2
    x =  tf.layers.conv2d_transpose(x, 512, (2, 2), (2, 2))
    # skip connection layer4
    x = tf.add(x, vgg_layer4_out)

    # 4x4
    x =  tf.layers.conv2d_transpose(x, 256, (2, 2), (2, 2))
    # skip connection layer3
    x = tf.add(x, vgg_layer3_out)

    # 8x8
    x =  tf.layers.conv2d_transpose(x, 128, (2, 2), (2, 2))

    # 16x16
    x =  tf.layers.conv2d_transpose(x, 64, (2, 2), (2, 2))

    # 32x32
    x =  tf.layers.conv2d_transpose(x, 32, (2, 2), (2, 2))

    # reduce to channels equivalent to number of classes
    x =  tf.layers.conv2d_transpose(x, num_classes, (1, 1), (1, 1))

    return x

tests.test_layers(layers)
# tests passed


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # calculate logits and loss
    # can later build a different loss metric, IOU, covered in semantic segmentation lesson
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label,
        logits=logits))

    # create optimization operations
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    return logits, train_op, loss

tests.test_optimize(optimize)
# tests passed

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # fix learning rate and keep_prob variables
    lr = 2e-4
    kp = 0.5

    # note: input image, correct label, keep_prob and learning rates are placeholders
    feed_dict={
        input_image: None,
        correct_label: None,
        learning_rate: lr,
        keep_prob: kp
    }

    # initialize variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        
        for images, gt_images in get_batches_fn(batch_size):
            feed_dict[input_image] = images
            feed_dict[correct_label] = gt_images
            _ , loss = sess.run([train_op, cross_entropy_loss], feed_dict)

        print('Epoch: {}, Current loss: {}'.format(epoch+1, loss))

tests.test_train_nn(train_nn)
#test passed

def run_download():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on. 
    #  https://www.cityscapes-dataset.com/
    # note: don't have this GPU, hardly could train on Kitti datasets on an Amazon P2 instance

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # to do later, after final project

        # Build NN using load_vgg, layers, and optimize function

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # create remaining placeholders
        correct_label = tf.placeholder(tf.float32, name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # get tensors for logits and train operation
        logits, train_op, loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        # define hyperparameters
        epochs = 300
        batch_size = 8

        # train neural network
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        # to do later, after final project

if __name__ == '__main__':
    run()