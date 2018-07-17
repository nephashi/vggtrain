import tensorflow as tf
import constant
IMAGE_SIZE = constant.IMAGE_SIZE
NUM_CLASSES = constant.NUM_CLASSES
BATCH_SIZE = constant.BATCH_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = constant.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

def input_source(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [96, 96, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

def generate_image_and_label_batch_with_queue(image, label,
                                              min_queue_examples, batch_size):
    num_preprocess_threads = 16
    images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 5 * batch_size,
            min_after_dequeue=min_queue_examples)
    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(labels, [batch_size])

def input(filename):
    imgsrc, labelsrc = input_source(filename);
    batch_img, batch_label = generate_image_and_label_batch_with_queue(imgsrc,
                    labelsrc, int(0.4 * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN), BATCH_SIZE)
    return batch_img, batch_label