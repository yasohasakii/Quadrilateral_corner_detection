import os
import sys
import argparse
import yaml
import urllib
import numpy as np
from PIL import Image
import tensorflow as tf
from hed.models.mobilenet_v1 import MobileNet_v1
from hed.models.vgg16 import Vgg16
from hed.utils._io import IO


class HEDTester():

    def __init__(self, config_file):

        self.io = IO()
        self.init = True

        try:
            pfile = open(config_file)
            self.cfgs = yaml.load(pfile)
            pfile.close()

        except Exception as err:

            self.io.print_error('Error reading config file {}, {}'.format(config_file), err)

    def setup(self, session):

        try:

            # self.model = Vgg16(self.cfgs, run='testing')
            self.model =  MobileNet_v1(self.cfgs)
            meta_model_file = os.path.join(self.cfgs['save_dir'], 'models/hed-model-{}'.format(self.cfgs['test_snapshot']))

            saver = tf.train.Saver()
            saver.restore(session, meta_model_file)

            self.io.print_info('Done restoring VGG-16 model from {}'.format(meta_model_file))

        except Exception as err:

            # self.io.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.io.print_error('Error setting up MobilNet_v1_model, {}'.format(err))
            self.init = False

    def run(self, session):

        if not self.init:
            return

        self.model.setup_testing(session)

        filepath = os.path.join(self.cfgs['download_path'], self.cfgs['testing']['list'])
        train_list = self.io.read_file_list(filepath)

        self.io.print_info('Writing PNGs at {}'.format(self.cfgs['test_output']))

        for idx, img in enumerate(train_list):

            test_filename = os.path.join(self.cfgs['download_path'], self.cfgs['testing']['dir'], img)
            im = self.fetch_image(test_filename)

            # edgemap = session.run(self.model.predictions, feed_dict={self.model.images: [im]})
            edgemap, coords_output = session.run(self.model.predictions, self.model.coord_pred, feed_dict={self.model.images:[im]})
            self.save_egdemaps(edgemap, idx)

            self.io.print_info('Done testing {}, {}, coords: {}'.format(test_filename, im.shape, coords_output))

    def save_egdemaps(self, em_maps, index):

        # Take the edge map from the network from side layers and fuse layer
        em_maps = [e[0] for e in em_maps]
        em_maps = em_maps + [np.mean(np.array(em_maps), axis=0)]

        for idx, em in enumerate(em_maps):

            em[em < self.cfgs['testing_threshold']] = 0.0

            em = 255.0 * (1.0 - em)
            em = np.tile(em, [1, 1, 3])

            em = Image.fromarray(np.uint8(em))
            em.save(os.path.join(self.cfgs['test_output'], 'testing-{}-{:03}.png'.format(index, idx)))

    def fetch_image(self, test_image):
        image = None

        # read from disk
        if os.path.exists(test_image):

            try:
                image = Image.open(test_image)
                image = image.resize((self.cfgs['testing']['image_width'], self.cfgs['testing']['image_height']))
                image = np.array(image, np.float32)
                image = self.colorize(image)

                image = image[:, :, self.cfgs['channel_swap']]
                image -= self.cfgs['mean_pixel_value']

            except Exception as err:
                print(self.io.print_error('[Testing] Error with image file {0} {1}'.format(test_image, err)))
                return None

        return image

    def colorize(self, image):

        # BW to 3 channel RGB image
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            image = np.tile(image, (1, 1, 3))
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        return image

if __name__ == '__main__':
    config_file = 'D:\Project\local_repository\holy-edge\hed\configs\hed.yaml'
    tester = HEDTester(config_file)
    filepath = os.path.join(tester.cfgs['download_path'], tester.cfgs['testing']['list'])
    train_list = tester.io.read_file_list(filepath)
    for idx, img in enumerate(train_list):
        test_filename = os.path.join(tester.cfgs['download_path'], tester.cfgs['testing']['dir'], img)
        im = tester.fetch_image(test_filename)
        tester.io.print_info('Done testing {}, {}'.format(test_filename, im.shape))
