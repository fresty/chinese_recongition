import tensorflow as tf
import input_data
import network_model
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


N_CLASS=3755
IMG_W=64
IMG_H=64
BATCH_SIZE=80
CAPACTIY=2000
MAX_STEP=50000
learning_rate=0.0001
log_dir='./log/'
def run_training():
    train_dir='/home/saverio_sun/project/chinese_rec_data/train'
    images,labels=input_data.read_files(train_dir)
    train_batch,train_label=input_data.get_batch(images,labels,IMG_W,IMG_H,BATCH_SIZE,CAPACTIY)
    train_logits=network_model.inference(train_batch,BATCH_SIZE,N_CLASS)
    train_loss=network_model.losses(train_logits,train_label)
    train_op=network_model.training(train_loss,learning_rate)
    train_acc=network_model.evaluation(logits=train_logits,labels=train_label)

    summary_op=tf.summary.merge_all()
    init=tf.global_variables_initializer()
    sess=tf.InteractiveSession()
    train_writer=tf.summary.FileWriter('./log/',sess.graph)
    saver=tf.train.Saver()
    sess.run(init)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc=sess.run([train_op,train_loss,train_acc])
            if step%100==0:
                print 'Step %d,train loss=%.2f, train accuracy=%.2f'%(step,tra_loss,tra_acc)
                summary_str=sess.run(summary_op)
                train_writer.add_summary(summary_str,global_step=step)
            if step%2000==0 or (step+1)==MAX_STEP:
                checkpoint_path=os.path.join(log_dir,'model.ckpt')
                saver.save(sess,save_path=checkpoint_path,global_step=step)
    except tf.errors.OutOfRangeError:
        print 'done'
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

def evaluate_one_pic(stri):
    image=Image.open(stri)
    #image=image.convert('L')
    plt.imshow(image)
    plt.show()
    image=image.resize([IMG_W,IMG_H])
    image_array=np.array(image)
    with tf.Graph().as_default():
        batch_size=1
        image=tf.cast(image_array,tf.float32)
        image=tf.image.rgb_to_grayscale(image)
        image=tf.reshape(image,[1,IMG_W,IMG_H,1])
        logit=network_model.inference(image,batch_size,N_CLASS)

        logit=tf.nn.softmax(logit)
        x=tf.placeholder(tf.float32,[IMG_W,IMG_H,3])
        saver=tf.train.Saver()
        with tf.Session() as sess:
            print 'reading checkpoint...'
            ckpt=tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_stop=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print 'load success'
            else:
                print 'No checkpoint'

            prediction =sess.run(logit,feed_dict={x:image_array})
            max_index=np.argmax(prediction)
            print max_index




if __name__=='__main__':
    #run_training()
    stri = '/home/saverio_sun/project/chinese_rec_data/test/03737/26695.png'
    evaluate_one_pic(stri)


