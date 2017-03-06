import urllib.request as url_req
import logging
import os, shutil
import hashlib
import zipfile
import tarfile
import logging
import re
import gzip
import sys
import pickle as cPickle

log = logging.getLogger(__name__)


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True


def verify_sha1(filename, sha1):
    data = open(filename, 'rb').read()
    if sha1 != hashlib.sha1(data).hexdigest():
        raise IOError('File \'%s\': invalid SHA-1 hash! You may want to delete '
                      'this corrupted file...' % filename)


def verify_md5(filename, md5sum, verbose=False):
    data = open(filename, 'rb').read()
    if verbose is True:
        print('Checking the MD5sum of the file %s' %filename)
    if md5sum != hashlib.md5(data).hexdigest():
        print('File \'%s\': invalid MD5 sum! You may want to delete '
                      'this corrupted file...' % filename)
        return False
    print('MD5 sum of the file is verified')
    return True


def download(url, output_filename, verbose=False):
    try:
        page = url_req.urlopen(url)
        if page.getcode() is not 200:
            if verbose is True:
                print('Tried to download data from %s and '
                      'got http response code %s' % (url, str(page.getcode())))
            log.warning('Tried to download data from %s and '
                        'got http response code %s', url, str(page.getcode()))
            return False
        else:
            if verbose is True:
                print('Downloading the CIFAR 10 dataset from %s to %s' %(url, output_filename))
            downloaded_file, _ = url_req.urlretrieve(url)
            if verbose is True:
                print('Moving the file to %s' %output_filename)
            shutil.move(downloaded_file, output_filename)
        return True
    except:
        if verbose is True:
            print('Error downloading data from %s to %s' % (url, output_filename))
        log.exception('Error downloading data from %s to %s', url, output_filename)
        return False


def extract(file_name, dest_dir=''):
    t = tarfile.open(file_name)
    if dest_dir == '':
        t.extractall()
        t.close()
    else:
        t.extractall(dest_dir)
        t.close()
    return True


def untar(source_filename, destination_dir='.'):
    source_filename = os.path.realpath(source_filename)
    destination_dir = os.path.realpath(destination_dir)
    log.debug('Unzipping tarball data from %s to %s', source_filename, destination_dir)
    try:
        with tarfile.open(source_filename) as tar:
            tar.extractall(destination_dir)
            tar.close()
            return True
    except:
        log.exception('Error unzipping tarball data from %s to %s', source_filename, destination_dir)
        return False


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding ='bytes')
    fo.close()
    return dict