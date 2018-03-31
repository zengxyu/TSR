import tensorflow as  tf
import TSRCnn
import TSRInput

W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2 = TSRCnn.initVariable()

#训练集数据
image, label = TSRInput.generate_image_and_label_from_images(data=TSRInput.train_data)
images, labels = TSRInput.generate_images_and_labels_batch(image=image, label=label, shuffle=True)



#测试集数据
image_test,label_test = TSRInput.generate_image_and_label_from_images(data=TSRInput.test_data)
images_test,labels_test = TSRInput.generate_images_and_labels_batch(image=image, label=label, shuffle=True)

#训练集上的准确度验证
# 用交叉熵来计算loss，训练集上
#卷积开始
y_conv = TSRCnn.deepnn3(images,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#测试集上准确度验证
y_conv_test = TSRCnn.deepnn3(images_test,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2)
correct_prediction_test = tf.equal(tf.argmax(y_conv_test, 1), tf.argmax(labels_test, 1))
accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"./Model1/model1.ckpt-5801")# 注意此处路径前添加"./"
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):  # 步数设置为6000
        acc = sess.run(accuracy_test)
        print('Accuracy:' , acc)
    coord.request_stop()
    coord.join(threads)
