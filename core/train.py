import os
import sys
import yaml
import argparse
import tensorflow as tf
from termcolor import colored

from hed.models.vgg16 import Vgg16
from hed.utils._io import IO
from hed.data.data_parser import DataParser
from hed.models.mobilenet_v1 import MobileNet_v1

class HEDTrainer():

    def __init__(self, config_file):

        self.io = IO()
        self.init = True

        try:
            pfile = open(config_file)
            self.cfgs = yaml.load(pfile)
            pfile.close()

        except Exception as err:

            print(('Error reading config file {}, {}'.format(config_file, err)))

    def setup(self):

        try:

            # self.model = Vgg16(self.cfgs)
            # self.io.print_info('Done initializing VGG-16 model')
            self.model = MobileNet_v1(self.cfgs)

            dirs = ['train', 'val', 'test', 'models']
            dirs = [os.path.join(self.cfgs['save_dir'] + '/{}'.format(d)) for d in dirs]
            _ = [os.makedirs(d) for d in dirs if not os.path.exists(d)]

        except Exception as err:

            # self.io.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.io.print_error('Error setting up MobileNet v1 model, {}'.format(err))
            self.init = False

    def run(self, session):

        if not self.init:
            return

        train_data = DataParser(self.cfgs)

        self.model.setup_training(session)

        opt = tf.train.AdamOptimizer(self.cfgs['optimizer_params']['learning_rate'])
        train = opt.minimize(self.model.loss)

        ckpt = tf.train.get_checkpoint_state(self.cfgs['save_dir'] + '/models')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reload model parameters..")
            self.model.restore(session, ckpt.model_checkpoint_path)
        elif not os.path.exists(self.cfgs['save_dir'] + '/models'):
                print("Created new model parameters..")
                session.run(tf.global_variables_initializer())
        else:
            session.run(tf.global_variables_initializer())

        for idx in range(self.cfgs['max_iterations']):

            im, em, _, cd, _ = train_data.get_training_batch()

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # _, summary, loss, side1,side2,side3,side4,side5,fuse = session.run([train, self.model.merged_summary, self.model.loss,
            #                                                                self.model.side_1,self.model.side_2, self.model.side_3, self.model.side_4, self.model.side_5,self.model.fuse],
            #                                feed_dict={self.model.images: im, self.model.edgemaps: em},
            #                                options=run_options,
            #                                run_metadata=run_metadata)
            # _, summary, loss, fuse, rn1, rn2, rn3, rn4, rn5, rn6, regressor = session.run([train, self.model.merged_summary, self.model.loss,self.model.fuse, self.model.rn1, self.model.rn2,
            #                                       self.model.rn3, self.model.rn4, self.model.rn5, self.model.rn6, self.model.regressor],
            #                                      feed_dict={self.model.images: im, self.model.edgemaps: em, self.model.coordinates: cd,self.model.is_train:True},
            #                                      options=run_options,run_metadata=run_metadata)
            _, summary, loss,  = session.run([train, self.model.merged_summary, self.model.loss],
					feed_dict={self.model.images: im, self.model.edgemaps: em, self.model.coordinates: cd, self.model.is_train: True},
                                             options=run_options,
                                             run_metadata=run_metadata)
            # print("rn1 shape:", rn1.shape)
            # print("rn2 shape:", rn2.shape)
            # print("rn3 shape:", rn3.shape)
            # print("rn4 shape:", rn4.shape)
            # print("rn5 shape:", rn5.shape)
            # print("rn6 shape:", rn6.shape)
            # print("regressor shape:", regressor.shape)
            # print("fuse shape:", fuse.shape)
            self.model.train_writer.add_run_metadata(run_metadata, 'step{:06}'.format(idx))
            self.model.train_writer.add_summary(summary, idx)

            self.io.print_info('[{}/{}] TRAINING loss : {}'.format(idx, self.cfgs['max_iterations'], loss))

            if idx % self.cfgs['save_interval'] == 0:

                saver = tf.train.Saver()
                saver.save(session, os.path.join(self.cfgs['save_dir'], 'models/hed-model'), global_step=idx)

            if idx % self.cfgs['val_interval'] == 0:

                im, em, _, cd, _ = train_data.get_validation_batch()

                summary, error, cd_error = session.run([self.model.merged_summary,
                                                        self.model.error,
                                                        self.model.cd_error
                                                        ],
                                                       feed_dict={self.model.images: im, self.model.edgemaps: em, self.model.coordinates: cd,self.model.is_train:False})

                self.model.val_writer.add_summary(summary, idx)
                self.io.print_info('[{}/{}] VALIDATION error : {}. CD error: {}'.format(idx, self.cfgs['max_iterations'], error, cd_error))

        self.model.train_writer.close()
