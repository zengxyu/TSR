# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import TSRInput
import TSRCnn

#训练集数据
image, label = TSRInput.generate_image_and_label_from_images(data=TSRInput.train_data)
images, labels = TSRInput.generate_images_and_labels_batch(image=image, label=label, shuffle=True)


#测试集数据
image_test,label_test = TSRInput.generate_image_and_label_from_images(data=TSRInput.test_data)
images_test,labels_test = TSRInput.generate_images_and_labels_batch(image=image, label=label, shuffle=True)

#网络所有的参数
W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2 = TSRCnn.initVariable()

# Build the graph for the deep net
#卷积开始
y_conv = TSRCnn.deepnn3(images,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2)

# 用交叉熵来计算loss，训练集上
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#测试集上准确度验证
y_conv_test = TSRCnn.deepnn3(images_test,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2)
correct_prediction_test = tf.equal(tf.argmax(y_conv_test, 1), tf.argmax(labels_test, 1))
accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))

# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
#保存模型
saver = tf.train.Saver(max_to_keep=3)


with tf.Session() as sess:
    # 运行初始化所有变量
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(TSRInput.LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(TSRInput.LOG_DIR + '/test')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    max_acc = 0
    for i in range(6000):#步数设置为6000
        if i % 50 == 0:  # 记录测试集的summary与accuracy
            summary, acc = sess.run([merged, accuracy_test])
            test_writer.add_summary(summary, i)
            #保存模型，保存精确度最高的三次
            if acc >= max_acc:
                max_acc = acc
                saver.save(sess, 'Model1/model1.ckpt', global_step=i + 1)
                print('Accuracy at step %s: %s' % (i, acc))
        else:  # 记录训练集的summary
            if i % 100 == 99:  # Record execution stats
                acc_train = sess.run(accuracy)
                summary, _ = sess.run([merged, train_step])
                train_writer.add_summary(summary, i)
                print('Accuracy at train step %s: %s' % (i, acc_train))
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step])
                train_writer.add_summary(summary, i)
        # saver.save(sess, "Model1/model1.ckpt")



    train_writer.close()
    test_writer.close()


    coord.request_stop()
    coord.join(threads)
