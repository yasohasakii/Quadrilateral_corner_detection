#!/usr/bin/python3
import os
import yaml
from pyunpack import Archive
from time import strftime, localtime
from termcolor import colored

#print("THE CURRUNT DIRECTORY:{} ".format(os.path.abspath('.')))

class IO():

    def __init__(self, log_dir=None):

        self.log_dir = log_dir

    def read_yaml_file(self, config_file):

        pfile = open(config_file)
        d = yaml.load(pfile)
        pfile.close()

        return d

    def download_data(self, filepath, outputdir):

        _, rar_file = os.path.split(filepath)
        rar_file = os.path.join(outputdir, rar_file)

        if not os.path.exists(rar_file):
            self.print_info('Downloading {} to {}'.format(filepath, rar_file))
            # _ = wget.download(filepath, out=outputdir)

        self.print_info('Decompressing {} to {}'.format(rar_file, outputdir))
        Archive(rar_file).extractall(outputdir)

    def print_info(self, info_string, quite=False):

        info = '[{0}][INFO] {1}'.format(self.get_local_time(), info_string)
        print(colored(info, 'green'))

    def print_warning(self, warning_string):

        warning = '[{0}][WARNING] {1}'.format(self.get_local_time(), warning_string)

        print(colored(warning, 'blue'))

    def print_error(self, error_string):

        error = '[{0}][ERROR] {1}'.format(self.get_local_time(), error_string)

        print(colored(error, 'red'))

    def get_local_time(self):

        return strftime("%d %b %Y %Hh%Mm%Ss", localtime())

    def read_file_list(self, filelist):

        pfile = open(filelist)
        filenames = pfile.readlines()
        pfile.close()

        filenames = [f.strip() for f in filenames]

        return filenames
    def read_file_list2(self, filelist):
        pfile = open(filelist)
        filenames = pfile.readlines()
        pfile.close()

        filenames = [f.strip() for f in filenames]

        return filenames

    def split_pair_names(self, filenames, base_dir):

        filenames = [c.split(' ') for c in filenames]
        filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]

        return filenames

    def split_triads(self, filenames, base_dir):
        filenames = [c.split(' ') for c in filenames]
        filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1]), c[2:4], c[4:]) for c in filenames]

        return filenames

if __name__ == "__main__":
    io = IO()
    file_list = '/mnt/hgfs/Project/local_repository/jd-edge/dataset_fk/train_pair.lst'
    base_dir = '/mnt/hgfs/Project/local_repository/jd-edge/dataset_fk/'
    filenames = io.read_file_list2(file_list)
    filenames_triads = io.split_triads(filenames, base_dir)
    print('closed')
    print()
