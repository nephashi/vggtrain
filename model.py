import tensorflow as tf
import constant
BATCH_SIZE = constant.BATCH_SIZE

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def network(image):
    '''
    Build VGG model
    :param images:[-1,96,96,3]
    :return: logit
    '''
    parameters = []
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable(name='weights', shape=[3,3,3,64],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(image, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[64],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv1)
    #[-1,96,96,64]
    print_activations(conv1)
    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable(name='weights', shape=[3,3,64,64],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv1, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[64],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv2)
    #[-1,96,96,64]
    print_activations(conv2)
    pool1 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1],
                           padding='SAME', name='pool1')
    #[-1,48,48,64]
    print_activations(pool1)
    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable(name='weights', shape=[3, 3, 64, 128],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[128],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv3)
    #[-1,48,48,128]
    print_activations(conv3)
    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable(name="weights", shape=[3,3,128,128],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[128],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv4)
    #[-1,48,48,128]
    print_activations(conv4)
    pool2 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1],
                           padding='SAME', name='pool2')
    #[-1,24,24,128]
    print_activations(pool2)
    with tf.variable_scope('conv5') as scope:
        kernel = tf.get_variable(name='weights', shape=[3,3,128,256],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[256],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv5)
    #[-1,24,24,256]
    print_activations(conv5)
    with tf.variable_scope('conv6') as scope:
        kernel = tf.get_variable(name='weights', shape=[3,3,256,256],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv5, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[256],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv6)
    #[-1,24,24,256]
    print_activations(conv6)
    with tf.variable_scope('conv7') as scope:
        kernel = tf.get_variable(name='weights', shape=[3,3,256,256],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv6, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[256],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv7 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv7)
    #[-1,24,24,256]
    print_activations(conv7)
    pool3 = tf.nn.max_pool(conv7, [1, 2, 2, 1], [1, 2, 2, 1],
                           padding='SAME', name='pool3')
    # [-1,12,12,256]
    print_activations(pool3)
    with tf.variable_scope('conv8') as scope:
        kernel = tf.get_variable(name='weights', shape=[3,3,256,512],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool3, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[512],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv8 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv8)
    # [-1,12,12,512]
    print_activations(conv8)
    with tf.variable_scope('conv9') as scope:
        kernel = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[512],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv9 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv9)
    # [-1,12,12,512]
    print_activations(conv9)
    with tf.variable_scope('conv10') as scope:
        kernel = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv9, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[512],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv10 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv10)
    # [-1,12,12,512]
    print_activations(conv10)
    pool4 = tf.nn.max_pool(conv10, [1, 2, 2, 1], [1, 2, 2, 1],
                           padding='SAME', name='pool4')
    # [-1,6,6,512]
    print_activations(pool4)
    with tf.variable_scope('conv11') as scope:
        kernel = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[512],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv11 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv11)
    # [-1,6,6,512]
    print_activations(conv11)
    with tf.variable_scope('conv12') as scope:
        kernel = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv11, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[512],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv12 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv12)
    # [-1,6,6,512]
    print_activations(conv12)
    with tf.variable_scope('conv13') as scope:
        kernel = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv12, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[512],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv13 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv13)
    # [-1,6,6,512]
    print_activations(conv13)
    pool5 = tf.nn.max_pool(conv13, [1, 2, 2, 1], [1, 2, 2, 1],
                           padding='SAME', name='pool5')
    # [-1,3,3,512]
    print_activations(pool5)
    with tf.variable_scope('fc1') as scope:
        # [-1, 3 * 3 * 512]
        pool5_flat = tf.reshape(pool5, [-1, 3 * 3 * 512])
        weights = tf.get_variable(name='weights', shape=[3 * 3 * 512, 4096],
                                  dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[4096],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        fc1 = tf.nn.relu(tf.matmul(pool5_flat, weights) + biases, name=scope.name)
        parameters += [weights, biases]
        _activation_summary(fc1)
    # [-1, 4096]
    print_activations(fc1)

    with tf.variable_scope('dp1') as scope:
        dp1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable(name='weights', shape=[4096, 4096],
                                  dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[4096],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        fc2 = tf.nn.relu(tf.matmul(dp1, weights) + biases, name=scope.name)
        parameters += [weights, biases]
        _activation_summary(fc2)
    # [-1, 4096]
    print_activations(fc2)
    with tf.variable_scope('dp2') as scope:
        dp2 = tf.nn.dropout(fc2, 0.5)
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable(name='weights', shape=[4096, 200],
                                  dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[200],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        logit = tf.add(tf.matmul(dp2, weights), biases, name=scope.name)
        parameters += [weights, biases]
        _activation_summary(logit)
    print_activations(logit)

    return logit

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return cross_entropy_mean

# img = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,96,96,3])
# network(img)