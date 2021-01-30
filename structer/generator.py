import math
import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size, tiled_dlatent, model_scale=18):
    if tiled_dlatent:
        low_dim_dlatent = tf.get_variable('learnable_dlatents',
            shape=(batch_size, 512),
            dtype='float32',
            initializer=tf.initializers.random_normal())
        # tf.tile(v,[a,b]) 按照一定的维度对张量进行复制填充， 若v=[[1],[2]] a=2,b=3 则结果为[[1,1],[2,2],[1,1],[2,2],[1,1],[2,2]]
        return tf.tile(tf.expand_dims(low_dim_dlatent, axis=1), [1, model_scale, 1])
    else:
        return tf.get_variable('learnable_dlatents',
            shape=(batch_size, model_scale, 512),
            dtype='float32',
            initializer=tf.initializers.random_normal()) # get_variable 函数用于创建一个新的变量或者获取已有变量


class Generator:
    def __init__(self, model, batch_size, clipping_threshold=2, tiled_dlatent=False, model_res=1024, randomize_noise=False):
        self.batch_size = batch_size
        self.tiled_dlatent=tiled_dlatent # 是否平铺
        self.model_scale = int(2*(math.log(model_res,2)-1)) # For example, 1024 -> 18

        if tiled_dlatent:
            self.initial_dlatents = np.zeros((self.batch_size, 512))
            model.components.synthesis.run(np.zeros((self.batch_size, self.model_scale, 512)),
                randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, tiled_dlatent=True),
                               partial(create_stub, batch_size=batch_size)],
                structure='fixed')
        else:
            self.initial_dlatents = np.zeros((self.batch_size, self.model_scale, 512))
            model.components.synthesis.run(self.initial_dlatents,
                randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, tiled_dlatent=False,model_scale=self.model_scale),
                               partial(create_stub, batch_size=batch_size)],
                structure='fixed')

        self.dlatent_avg_def = model.get_var('dlatent_avg')
        self.reset_dlatent_avg()
        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
        self.set_dlatents(self.initial_dlatents) # 执行赋值过程，initial_dlatents 的值复制给self.dlatent_variable

        try:
            self.generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0')
        except KeyError:
            # If we loaded only Gs and didn't load G or D, then scope "G_synthesis_1" won't exist in the graph.
            self.generator_output = self.graph.get_tensor_by_name('G_synthesis/_Run/concat:0')
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8) # 安全的类型转换

        # Implement stochastic clipping similar to what is described in https://arxiv.org/abs/1702.04782
        # (Slightly different in that the latent space is normal gaussian here and was uniform in [-1, 1] in that paper,
        # so we clip any vector components outside of [-2, 2]. It seems fine, but I haven't done an ablation check.)
        # print(self.dlatent_variable.shape) (1,18,512)
        clipping_mask = tf.math.logical_or(self.dlatent_variable > clipping_threshold, self.dlatent_variable < -clipping_threshold)
        clipped_values = tf.where(clipping_mask, tf.random_normal(shape=self.dlatent_variable.shape), self.dlatent_variable)
        # tf.where(a,b,c)函数：当a输出结果为true时，tf.where(a,b,c)函数会选择b值输出。
        self.stochastic_clip_op = tf.assign(self.dlatent_variable, clipped_values)

    def set_dlatents(self, dlatents):
        if self.tiled_dlatent:
            if (dlatents.shape != (self.batch_size, 512)) and (dlatents.shape[1] != 512):
                dlatents = np.mean(dlatents, axis=1)
            if (dlatents.shape != (self.batch_size, 512)):
                dlatents = np.vstack([dlatents, np.zeros((self.batch_size-dlatents.shape[0], 512))])
            assert (dlatents.shape == (self.batch_size, 512))
        else: # 如果不平铺
            if (dlatents.shape[1] > self.model_scale):
                dlatents = dlatents[:,:self.model_scale,:]
            if (dlatents.shape != (self.batch_size, self.model_scale, 512)):
                dlatents = np.vstack([dlatents, np.zeros((self.batch_size-dlatents.shape[0], self.model_scale, 512))])
            assert (dlatents.shape == (self.batch_size, self.model_scale, 512))
        self.sess.run(tf.assign(self.dlatent_variable, dlatents)) # 执行赋值过程，将dlatents 的值复制给self.dlatent_variable

    def reset_dlatent_avg(self):
        self.dlatent_avg = self.dlatent_avg_def

    def generate_images(self, dlatents=None):
        if dlatents:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)