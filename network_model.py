import tensorflow as tf

def inference(images,batch_size,n_class):

    with tf.variable_scope('conv1') as scope :
        weights=tf.get_variable('weights',
                                [3,3,1,16],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias=tf.get_variable('bias',
                             [16],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding='SAME')
        conv1=tf.nn.relu(tf.nn.bias_add(conv,bias))

    with tf.variable_scope('pool1') as scope:
        pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


    with tf.variable_scope('conv2') as scope:
        weights=tf.get_variable('weights',
                                [3,3,16,32],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias=tf.get_variable('bias',
                             [32],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(pool1,weights,strides=[1,1,1,1],padding='SAME')
        conv2=tf.nn.relu(tf.nn.bias_add(conv,bias))

    with tf.variable_scope('pool2') as scope:
        pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


    with tf.variable_scope('fc3') as scope:
        reshape=tf.reshape(pool2,[batch_size,-1])
        in_num=reshape.get_shape()[1].value
        weights=tf.get_variable('weights',
                                [in_num,1024],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        bias=tf.get_variable('bias',
                             [1024],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
        fc3=tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape,weights),bias))

    with tf.variable_scope('fc4') as scope :
        weights=tf.get_variable('weights',
                                [1024,n_class],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        bias=tf.get_variable('bias',
                             [n_class],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
        fc4=tf.nn.bias_add(tf.matmul(fc3,weights),bias)

    return fc4

def losses(logits,labels):
    with tf.name_scope('loss_layer'):
        cross_entray=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        loss=tf.reduce_mean(cross_entray)
        tf.summary.scalar("loss",loss)
    return loss

def training(loss,learning_rate):
    with tf.variable_scope('optimizer') as scope:
        optimizer=tf.train.AdamOptimizer(learning_rate)
        global_step=tf.Variable(0,name='global_step',trainable=False)
        train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct=tf.nn.in_top_k(logits,labels,1)
        accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
        tf.summary.scalar("acc",accuracy)
    return accuracy


