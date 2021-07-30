from ops import *
from utils import *
import time
import glob
from tensorflow.contrib.data import batch_and_drop_remainder
import scipy.io as sio
import random

class PS(object) :
    def __init__(self, sess, args):
        self.model_name = 'my_PS'
        self.sess = sess
        self.N = args.N
        self.NFILTER = args.NFILTER
        self.NCHANNEL = args.NCHANNEL
        self.image_h = args.image_h
        self.image_w = args.image_w
        self.NV = args.NV
        self.batch_size = args.batch_size
        self.num_cpu_cores = args.num_cpu_cores
        self.checkpoint_dir = args.checkpoint_dir
        self.scope_idx = args.scope_idx

    #parser for [ball,cat,pot1,pot2,buddha,goblet,reading,harvest,cow]
    def parser(self, record):
        features = tf.parse_single_example(
            record,
            features={
                'image': tf.FixedLenFeature([self.N*self.NCHANNEL*self.image_h*self.image_w],tf.float32),
                'lable': tf.FixedLenFeature([self.image_h*self.image_w*3],tf.float32),
                'mask': tf.FixedLenFeature([self.image_h*self.image_w*3],tf.float32)
            }
        )
        image=tf.reshape(features['image'], [self.N, self.NCHANNEL, self.image_h*self.image_w])
        lable=tf.reshape(features['lable'], [self.image_h, self.image_w, 3])
        mask=tf.reshape(features['mask'], [self.image_h, self.image_w, 3])
        return image, lable, mask
    #parser for [bear]
    def parser_bear(self, record):
        features = tf.parse_single_example(
            record,
            features={
                'image': tf.FixedLenFeature([(self.N-20)*self.NCHANNEL*self.image_h*self.image_w],tf.float32),
                'lable': tf.FixedLenFeature([self.image_h*self.image_w*3],tf.float32),
                'mask': tf.FixedLenFeature([self.image_h*self.image_w*3],tf.float32)
            }
        )
        image=tf.reshape(features['image'], [(self.N-20), self.NCHANNEL, self.image_h*self.image_w])
        lable=tf.reshape(features['lable'], [self.image_h, self.image_w, 3])
        mask=tf.reshape(features['mask'], [self.image_h, self.image_w, 3])
        return image, lable, mask

    ##################################################################################
    ##################################################################################
    def cond(self, a, y, q1, q2, r):
        return tf.less(a, self.NV)
    def body(self, a, tensor_array, q1, q2, r):
        temp = 2 * r * q1 - q2
        tensor_array = tensor_array.write(a, temp)
        q2 = q1
        q1 = temp
        a += 1
        return a, tensor_array, q1, q2, r
    def func1(self, R, weights2, flag):
        if flag == 'bear':
            N = self.N-20
        else:
            N = self.N
    
        tensor_array = tf.TensorArray(size=self.NV,dtype=tf.float32)
        q2 = tf.ones([N * self.image_h * self.image_w, 1], dtype=tf.float32)
        tensor_array = tensor_array.write(0, q2)
        tensor_array = tensor_array.write(1, R)
        q1 = R
        a = tf.constant(2)
        a, tensor_array, q1, q2, R = tf.while_loop(self.cond, self.body, [a, tensor_array, q1, q2, R])
        tensor_array = tensor_array.stack()
        tensor_array = tf.squeeze(tensor_array)
        tensor_array = tf.transpose(tensor_array, [1, 0])
        yout = tf.matmul(tensor_array, weights2)
        return yout
    def funcx(self, input,weights1_1, weights1_2, weights1_3, weights2, flag):  
        if flag == 'bear':
            N = self.N-20
        else:
            N = self.N
        x1 = tf.transpose(input, [0,2,1])
        x1 = tf.reshape(x1, [N * self.image_h * self.image_w, self.NCHANNEL]) 
        x2 = tf.tile(input, [1, self.NFILTER, 1]) 
        x2 = tf.transpose(x2, [0, 2, 1])  
        x2 = tf.reshape(x2, [N * self.image_h * self.image_w, self.NCHANNEL*self.NFILTER])

        r = tf.matmul(x1, weights1_1)
        r = tf.nn.leaky_relu(r, 0.2)
        r = tf.matmul(r, weights1_2)
        r = tf.nn.leaky_relu(r, 0.2)
        r = tf.matmul(r, weights1_3)
        r = tf.nn.leaky_relu(r, 0.2)
        R = tanh(r) 

        T3 = self.func1(R, weights2, flag) 
        T3 = tf.multiply(T3, x2)  
        T3 = tf.reshape(T3, [N * self.image_h * self.image_w, self.NFILTER, self.NCHANNEL])
        T3 = tf.reshape(T3, [N, self.image_h * self.image_w, self.NFILTER, self.NCHANNEL]) 
        
        T3 = tf.transpose(T3, [2, 1, 0, 3])
        T3 = tf.reduce_sum(T3, 3) 
        T5_max = tf.reduce_max(T3, 2, keep_dims=True)
        T5_mean = tf.reduce_mean(T3, 2, keep_dims=True)
        T5_merge = tf.concat([T5_max,T5_mean],-1)
        T5_merge = tf.reshape(T5_merge, [self.NFILTER, self.image_h, self.image_w, 2])
        T5_merge = tf.transpose(T5_merge, [1, 2, 0, 3])
        T5_merge = tf.reshape(T5_merge, [self.image_h, self.image_w, self.NFILTER*2])
        return T5_merge

    def inference(self, input_tensor, reuse):
        with tf.variable_scope("way_"+self.scope_idx):
            x_64=conv(input_tensor, 64, kernel_size=1, scope="x_64", use_relu=True, reuse=reuse)
            x_64=resblock(x_64, 64, kernel_size=1, scope="res_64", reuse=reuse)
            x_64=lrelu(x_64)
            x_128=conv(x_64, 128, kernel_size=1, scope="x_128", use_relu=True, reuse=reuse)
            x_128=resblock(x_128, 128, kernel_size=1, scope="res_128", reuse=reuse)
            x_128=lrelu(x_128)
            
            x_256=conv(x_128, 256, kernel_size=3, scope="x_256", use_relu=True, reuse=reuse)
            x_64_2=conv(x_256, 64, kernel_size=1, scope="x_64_2", use_relu=True, reuse=reuse)
            x_64_2=resblock(x_64_2, 64, kernel_size=1, scope="res_64_2", reuse=reuse)
            x_64_2=lrelu(x_64_2)
            
            x_256_tt=conv(x_128, 256, kernel_size=1, scope="x_256_tt", use_relu=True, reuse=reuse)
            x_64_2_tt=conv(x_256_tt, 64, kernel_size=1, scope="x_64_2_tt", use_relu=True, reuse=reuse)
            x_64_2_tt=resblock(x_64_2_tt, 64, kernel_size=1, scope="res_64_2_tt", reuse=reuse)
            x_64_2_tt=lrelu(x_64_2_tt)
            
            x_out = tf.concat([x_64_2,x_64_2_tt],axis=-1)
            return x_out

    def build_model(self):
        with tf.variable_scope('weights1_1'):
            weights1_1 = get_weight_variable([self.NCHANNEL, 10])
        with tf.variable_scope('weights1_2'):
            weights1_2 = get_weight_variable([10, 10])
        with tf.variable_scope('weights1_3'):
            weights1_3 = get_weight_variable([10, 1])                                            # input_mask    [None,h,w,3]  0/1  
        with tf.variable_scope('weights2'):
            weights2 = get_weight_variable(
                [self.NV, self.NCHANNEL * self.NFILTER]
            )
        #build inference pipeline for [ball,cat,pot1,pot2,buddha,goblet,reading,harvest,cow]
        input_files = []
        for obj in ['ball','cat','pot1','pot2','buddha','goblet','reading','harvest','cow']:
            input_files += glob.glob('test_data/'+'data.tfrecords_'+obj+'PNG')
        dataset=tf.data.TFRecordDataset(input_files)
        dataset=dataset.map(self.parser,num_parallel_calls=self.num_cpu_cores).apply(batch_and_drop_remainder(self.batch_size)).repeat() 
        iterator = dataset.make_one_shot_iterator()
        self.image, self.label, self.mask = iterator.get_next()
        self.label = l2_norm(self.label)*self.mask       
        x1 = tf.map_fn(fn=lambda inp: self.funcx(inp, weights1_1, weights1_2, weights1_3, weights2, ''), elems=self.image, dtype=tf.float32)
        self.y = self.inference(x1,reuse=False)
        self.y = get_nomal_map(self.y,self.mask,"y"+self.scope_idx,reuse=False)
        self.angle_error,self.pixel_num = angle_loss_acos(self.mask, self.y, self.label)
        self.error_map = error_map(self.mask, self.y, self.label)
        
        #build inference pipeline for [bear]        
        input_files_bear = []
        for obj in ['bear']:
            input_files_bear += glob.glob('test_data/'+'data.tfrecords_'+obj+'PNG')
        dataset_bear=tf.data.TFRecordDataset(input_files_bear)
        dataset_bear=dataset_bear.map(self.parser_bear,num_parallel_calls=self.num_cpu_cores).apply(batch_and_drop_remainder(self.batch_size)).repeat() 
        iterator_bear = dataset_bear.make_one_shot_iterator()
        self.image_bear, self.label_bear, self.mask_bear = iterator_bear.get_next()
        self.label_bear = l2_norm(self.label_bear)*self.mask_bear        
        x1_bear = tf.map_fn(fn=lambda inp: self.funcx(inp, weights1_1, weights1_2, weights1_3, weights2, 'bear'), elems=self.image_bear, dtype=tf.float32)
        self.y_bear = self.inference(x1_bear,reuse=True)
        self.y_bear = get_nomal_map(self.y_bear,self.mask_bear,"y"+self.scope_idx,reuse=True)
        self.angle_error_bear,self.pixel_num_bear = angle_loss_acos(self.mask_bear, self.y_bear, self.label_bear)
        self.error_map_bear = error_map(self.mask_bear, self.y_bear, self.label_bear)

        
        
    def cal(self):   
        obj_list = ['ball','cat','pot1','pot2','buddha','goblet','reading','harvest','cow','bear']
        result_list = []
        for idx in range(0,len(obj_list)):
            sum_error = 0
            sum_pixel = 0
            est_list = []
            for ii in range(16):
                if obj_list[idx]=='bear':
                    pixel_num, error, est, Err = self.sess.run([self.pixel_num_bear,self.angle_error_bear,self.y_bear,self.error_map_bear])
                else:
                    pixel_num, error, est, Err = self.sess.run([self.pixel_num,self.angle_error,self.y,self.error_map])
                est_list.append(est)
                if pixel_num>0:
                    sum_error += error*pixel_num
                    sum_pixel += pixel_num
            mean_error = sum_error/sum_pixel
            print('%s  %.2f' %(obj_list[idx],mean_error))
            result_list.append(mean_error)
            EST = np.concatenate( [np.concatenate([est_list[0][0,:,:,:],est_list[1][0,:,:,:],est_list[2][0,:,:,:],est_list[3][0,:,:,:]],axis=1), 
                                   np.concatenate([est_list[4][0,:,:,:],est_list[5][0,:,:,:],est_list[6][0,:,:,:],est_list[7][0,:,:,:]],axis=1), 
                                   np.concatenate([est_list[8][0,:,:,:],est_list[9][0,:,:,:],est_list[10][0,:,:,:],est_list[11][0,:,:,:]],axis=1), 
                                   np.concatenate([est_list[12][0,:,:,:],est_list[13][0,:,:,:],est_list[14][0,:,:,:],est_list[15][0,:,:,:]],axis=1)], axis=0)
            cv2.imwrite('result/%s_%.2f.png' %(obj_list[idx],mean_error), (EST[:,:,::-1]+1)*127.5)
        return np.sum(result_list)

    @property
    def model_dir(self):
        return "{}".format(self.model_name)
            
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    def test(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print(self.cal())
            
