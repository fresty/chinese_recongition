import tensorflow as tf
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt

def read_files(data_str):
    image=[]
    label=[]
    for root,folder,files in os.walk(data_str):
        image+=[os.path.join(root,name) for name in files]
        # for name in files:
        #     image.append(os.path.join(root,name))
        #     kind=name.split('.')[-3]
        #     if kind=='cat':
        #         label.append(0)
        #     else:
        #         label.append(1)

    label=[int(image_path.split('/')[-2]) for image_path in image]
    temp=np.array([image,label])
    temp=np.transpose(temp)
    np.random.shuffle(temp)

    image_list=list(temp[:,0])
    label_list=list(temp[:,1])
    label_list=[int(i) for i in label_list]
    return image_list,label_list

def get_batch(images,labels,img_w,img_h,batch_size,cap):
    images=tf.cast(images,tf.string)
    labels=tf.cast(labels,tf.int32)
    quener=tf.train.slice_input_producer([images,labels])
    labels_list=quener[1]
    images_contents=tf.read_file(quener[0])
    images_list=tf.image.decode_png(images_contents,channels=3)
    #images_list=tf.image.resize_images(images_list,[img_w,img_h])
    images_list=tf.image.rgb_to_grayscale(images_list)
    images_list=tf.image.resize_image_with_crop_or_pad(images_list,img_w,img_h)
    images_list=tf.image.per_image_standardization(images_list)
    image_batch,label_batch=tf.train.batch([images_list,labels_list],batch_size,64,capacity=cap)
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch

def test():
    #train_dir='/home/saverio_sun/project/chinese_rec_data/little_train'
    train_dir = '/home/saverio_sun/project/learning_tensorflow/train'
    IMG_W=208
    IMG_H=208
    BATCH_SIZE=3
    CAP=256
    image_list,label_list=read_files(train_dir)
    image_batch,label_batch=get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAP)
    with tf.Session() as sess:
        i=0
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i<1:
                img,label=sess.run([image_batch,label_batch])
                for j in np.arange(BATCH_SIZE):
                    print 'label: %d'%label[j]
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i+=1
        except tf.errors.OutOfRangeError:
            print 'done!'
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__=='__main__':
    test()


