import urllib.request as url_req
import os, shutil
import hashlib
import tarfile
import logging
import sys
import pickle as cPickle
import zipfile
import glob

log = logging.getLogger(__name__)


def mkdir_p(path, verbose=False):
    '''

    :param path:
    :param verbose:
    :return:
    '''
    if verbose is True:
        print('Creating the directory \'%s\'' %path)
    # Check if directory exists else make one
    if not os.path.exists(path):
        os.makedirs(path)
    return True


def delete_all_files_in_dir(directory):
    if os.path.exists(directory):
        files = glob.glob(directory+'*')
        for file in files:
            os.remove(file)


def verify_sha1(filename, sha1, verbose=False):
    '''

    :param filename:
    :param sha1:
    :param verbose:
    :return:
    '''
    if verbose is True:
        print('Verifying the SHA1 (%s) of the file: %s' %(sha1,filename))
    data = open(filename, 'rb').read()
    if sha1 != hashlib.sha1(data).hexdigest():
        raise IOError('File \'%s\': invalid SHA-1 hash! You may want to delete '
                      'this corrupted file...' % filename)
    else:
        print('SHA1 of the file: %s is verified' %filename)
        return True


def verify_md5(filename, md5sum, verbose=False):
    '''

    :param filename:
    :param md5sum:
    :param verbose:
    :return:
    '''
    if verbose is True:
        print('Verifying the MD5 (%s) of the file: %s' %(md5sum,filename))
    data = open(filename, 'rb').read()
    if md5sum != hashlib.md5(data).hexdigest():
        raise IOError('File \'%s\': invalid MD5 sum! You may want to delete '
                      'this corrupted file...' % filename)
    else:
        print('MD5sum of the file: %s is verified' %filename)
        return True


def download(url, output_filename, verbose=False):
    '''

    :param url:
    :param output_filename:
    :param verbose:
    :return:
    '''
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
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s. Progress: %.5f%%' % (output_filename,
                                                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            print('Downloading the CIFAR 10 dataset from %s to %s' %(url, output_filename))
            downloaded_file, _ = url_req.urlretrieve(url=url, filename=output_filename, reporthook=_progress)
            print()
            if verbose is True:
                print('Moving the file to %s' %output_filename)
            shutil.move(downloaded_file, output_filename)
        return True
    except:
        raise IOError('Error downloading data from %s to %s', url, output_filename)
        return False


def extract(file_name, dest_dir='', verbose=False):
    '''

    :param file_name:
    :param dest_dir:
    :param verbose:
    :return:
    '''
    if file_name.endswith(('.tar.gz', '.tgz')):
        t = tarfile.open(name=file_name, mode='r:gz')
        if dest_dir == '':
            if verbose is True:
                print('Extracting tar file: %s' % (file_name))
            t.extractall()
            t.close()
        else:
            if verbose is True:
                print('Extracting tar file: %s to %s' % (file_name, dest_dir))
            t.extractall(dest_dir)
            t.close()
    if file_name.endswith('.zip'):
        z = zipfile.Zipfile(file=file_name, mode='r')
        if dest_dir == '':
            if verbose is True:
                print('Extracting zip file: %s' % (file_name))
            z.extractall()
            z.close()
        else:
            if verbose is True:
                print('Extracting zip file: %s to %s' % (file_name, dest_dir))
            z.extractall(dest_dir)
            z.close()
    return True


def untar(source_filename, destination_dir='.', verbose=False):
    '''

    :param source_filename:
    :param destination_dir:
    :param verbose:
    :return:
    '''
    source_filename = os.path.realpath(source_filename)
    destination_dir = os.path.realpath(destination_dir)
    if verbose is True:
        print('Unzipping tarball data from %s to %s', source_filename, destination_dir)
    try:
        with tarfile.open(source_filename) as tar:
            tar.extractall(destination_dir)
            tar.close()
            return True
    except:
        raise IOError('Error unzipping tarball data from %s to %s', source_filename, destination_dir)
        return False


def unpickle(file, verbose=False):
    '''

    :param file:
    :param verbose:
    :return:
    '''
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding ='bytes')
    fo.close()
    return dict