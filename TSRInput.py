import os
import tensorflow as tf

DATA_DIR = "data3"
LOG_DIR = "log_dir"
IMAGE_WIDTH = 48#图片的宽
IMAGE_HEIGHT = 48#图片的高
IMAGE_DEPTH = 3
LABEL_BYTES = 1#
NUM_CLASSES = 43#类别数量
NUM_EXAMPLES_FOR_TRAIN = 35288#训练的样本数量
BATCH_SIZE = 350#每一批次处理的样本数量


train_data = "train_data.csv"
test_data = "test_data.csv"

def generate_image_and_label():
    filenames = [os.path.join(DATA_DIR, 'mnist_batch%d.csv' % i)
                 for i in range(6)]
    filename_queue = tf.train.string_input_producer(filenames)
    image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH*IMAGE_DEPTH
    record_bytes = image_bytes + LABEL_BYTES
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_default = []
    for i in range(record_bytes):
        record_default.append([1])
    records= tf.decode_csv(value,record_defaults=record_default)
    image = tf.reshape(tf.cast(tf.strided_slice(records, [0], [image_bytes]), tf.float32),
                       [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    # float_image = tf.image.per_image_standardization(image)
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    image = tf.transpose(image, [2, 1, 0])
    # tf.image.resize_images(image,[IMAGE_HEIGHT,IMAGE_WIDTH,1])
    label = tf.cast(tf.strided_slice(records, [image_bytes], [record_bytes]), tf.int32)
    label.set_shape([1])
    return image,label
    # print(records)
    # return records
def generate_image_and_label_from_images(data):
    #eval ，是不是训练集数据
    # if not eval:
    filename = os.path.join(DATA_DIR, data)
    # else:
    #     filename = os.path.join(DATA_DIR, "train_data.csv")
    with open(filename) as fid:
        content = fid.read()
    content = content.split("\n")
    content = content[:-1]
    valuequeue = tf.train.string_input_producer(content,shuffle=True)
    value = valuequeue.dequeue()
    image_path,label = tf.decode_csv(records=value,record_defaults=[["string"],[""]])
    label = tf.string_to_number(label,tf.int32)
    imageContent = tf.read_file(image_path)#读图片文件
    image = tf.image.decode_png(imageContent,channels=3,dtype=tf.uint8)
    image = tf.cast(image,tf.float32)
    image = tf.image.resize_images(image,[48,48])
    reshaped_image = tf.reshape(image,[IMAGE_DEPTH,IMAGE_HEIGHT,IMAGE_WIDTH])

    #图像预处理#
    #随意裁剪
    distorted_image = tf.random_crop(reshaped_image,[IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH])

    #随意增加亮度
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)

    #随意增加对比度
    distorted_image - tf.image.random_contrast(distorted_image,lower=0.2,upper=1.0)

    float_image = tf.image.per_image_standardization(distorted_image)


    # image = tf.transpose(image,[1,2,0])
    float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_DEPTH])

    return image,label


# 产生图片队列
def generate_images_and_labels_batch(image, label, shuffle):
    label = tf.reshape(tf.one_hot(label,depth=NUM_CLASSES,axis=0),[NUM_CLASSES])
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(min_fraction_of_examples_in_queue * NUM_EXAMPLES_FOR_TRAIN)
    num_preprocess_threads = 16
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=BATCH_SIZE,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=BATCH_SIZE,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples
        )
    return images, labels