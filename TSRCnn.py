import tensorflow as tf
import TSRInput
#用于构建卷积神经网络
# 输入图片是28*28*3
def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    # shape的第一个参数-1表示维度根据后面的维度计算得到，保持总的数据量不变
    x_image = tf.reshape(x, [-1, 28, 28, 3])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    # [5, 5, 1, 32]中第一个参数表示卷积核的高度，第二个表示卷积核的宽度，
    # 第三个参数表示输入channel的数量，因为MNIST都是灰度图，所以只有一个channel，
    # RGB图像有3个channel，第四个参数表示输出的channel数量
    W_conv1 = weight_variable([5, 5, 3, 32])
    # 定义bias变量，是一个长度为32的向量，每一维被初始化为0.1
    b_conv1 = bias_variable([32])
    # tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,
    # data_format=None,name=None)函数用来做卷积操作
    # input参数是一个Tensor，这里是一批原始图片，filter是这一层的所有权重
    # strides参数指滑动窗口在input参数各个维度滑动的步长，通常每次只滑动一个像素
    # padding有两个值："SAME"或"VALID"，
    # 具体的意义可参考：https://www.tensorflow.org/api_guides/python/nn#convolution
    # tf.nn.relu函数的全称为：Rectified Linear Unit，这是一个非常简单的函数，
    # 即：f(x) = max(0, x)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    # tf.nn.max_pool(value,ksize,strides,padding,data_format='NHWC',name=None)
    # 第一个参数value是输入数据
    # 第二个参数ksize是滑动窗口在第一个参数各个维度上的大小，长度至少为4
    # 第三个参数strides是滑动窗口在各个维度上的步长，和第二个参数一样，长度至少为4
    # 第四个参数padding与tf.nn.conv2d方法中的一样
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    # 构造第二层的权重，在这一层有32个输入channel，总共产生64个输出，参数多了起来
    W_conv2 = weight_variable([5, 5, 32, 64])
    # 因为每个channel产生64个输出，这里就需要64个bias参数了
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    # 全连接层，一共有1024个神经元
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    # 每个神经元需要一个bias
    b_fc1 = bias_variable([1024])

    # h_pool2的shape为[batch,7,7,64],将其reshape为[-1, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    # keep_prob = tf.placeholder(tf.float32)
    # dropout是将某一些神经元的输出变为0，这是为了防止过拟合
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, TSRInput.NUM_CLASSES])
    b_fc2 = bias_variable([TSRInput.NUM_CLASSES])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

#三层卷积层，二层全连接层的网络
def deepnn2(x):

    x_image = tf.reshape(x, [-1, 48, 48, 3])

    #第一层卷基层
    W_conv1 = weight_variable([5, 5, 3, 100])
    variable_summaries(W_conv1)

    b_conv1 = bias_variable([100])
    variable_summaries(b_conv1)

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    tf.summary.histogram('h_conv1', h_conv1)
    #第一层池化层
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积层
    W_conv2 = weight_variable([3, 3, 100, 150])
    b_conv2 = bias_variable([150])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #第三层卷积层
    W_conv3 = weight_variable([3, 3, 150, 250])
    # 因为每个channel产生64个输出，这里就需要64个bias参数了
    b_conv3 = bias_variable([250])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #第一层全连接层
    W_fc1 = weight_variable([4 * 4 * 250, 300])
    b_fc1 = bias_variable([300])
    h_pool2_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 250])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # dropout是将某一些神经元的输出变为0，这是为了防止过拟合
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    #第二层全连接层
    W_fc2 = weight_variable([300, TSRInput.NUM_CLASSES])
    b_fc2 = bias_variable([TSRInput.NUM_CLASSES])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

#初始化变量
def initVariable():
    W_conv1 = weight_variable([3, 3, 3, 32])
    # 定义bias变量，是一个长度为32的向量，每一维被初始化为0.1
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([3, 3, 32, 64])
    # 因为每个channel产生64个输出，这里就需要64个bias参数了
    b_conv2 = bias_variable([64])

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # 全连接层，一共有1024个神经元
    W_fc1 = weight_variable([10 * 10 * 64, 1024])
    # 每个神经元需要一个bias
    b_fc1 = bias_variable([1024])

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, TSRInput.NUM_CLASSES])
    b_fc2 = bias_variable([TSRInput.NUM_CLASSES])
    #可视化
    variable_summaries(W_conv1)
    variable_summaries(b_conv1)
    variable_summaries(W_conv2)
    variable_summaries(b_conv2)
    variable_summaries(W_fc1)
    variable_summaries(b_fc1)
    variable_summaries(W_fc2)
    variable_summaries(b_fc2)

    return W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2

#两层全连接层，两层池化层
def deepnn3(x,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2):

    x_image = tf.reshape(x, [-1, 48, 48, 3])

    #第一层卷积层
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #第一层池化层
    h_pool1 = max_pool_2x2(h_conv1)

    #第二层卷积层
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #第二层池化层
    h_pool2 = max_pool_2x2(h_conv2)

    #第一层全连接层
    # h_pool2的shape为[batch,7,7,64],将其reshape为[-1, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # dropout是将某一些神经元的输出变为0，这是为了防止过拟合
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    #第二层全连接层
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # return y_conv, keep_prob
    return y_conv

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)