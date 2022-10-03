"""@package file_utils Package for file utilities
Import necessary libraries
"""
import urllib.request as url_req
import os, shutil, hashlib, tarfile, sys
import pickle as cPickle
import zipfile, glob, logging

log = logging.getLogger(__name__)


def mkdir_p(path, verbose=False):
    """
    Creates a directory as declared by the path
    :param path: Directory to be created. Reative/Absolute path can be give
    :param verbose: If log information is to be printed on console
    :return: True if directory creation is successful else False
    """
    if verbose is True:
        print('Creating the directory \'%s\'' %path)
    # Check if directory exists else make one
    if not os.path.exists(path):
        os.makedirs(path)
    return True


def rm_rf(directory):
    """
    Delete all files in a directory
    :param directory: Name of directory in which all the files have to be deleted
    :return: True if deletion of all the files was successful else False 
    """
    if os.path.exists(directory):
        files = glob.glob(directory+'*')
        for file in files:
            os.remove(file)
    return True


def rm(filename):
    """
    Delete a particular file if it exists
    :param filename: Name of file which is to be deleted
    :return: True if deletion of the file was successful else False 
    """
    if os.path.exists(filename):
        os.remove(filename)
    return True


def verify_sha1(filename, sha1, verbose=False):
    """
    Verify the SHA1 of the file 
    :param filename: Name of the file whose MD5 is to be verified
    :param md5sum: Actual MD5 value of the file
    :param verbose: If log information is to be printed onto screen
    :return:
    """
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
    """
    Verify the MD5sum of the file 
    :param filename: Name of the file whose MD5 is to be verified
    :param md5sum: Actual MD5 value of the file
    :param verbose: If log information is to be printed onto screen
    :return:
    """
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
    """
    Download a file from the web
    :param url: URL of the file
    :param output_filename: Filename to which the downloadable file is to be written
    :param verbose: If log information is to be printed onto screen
    :return:
    """
    try:
        page = url_req.urlopen(url)
        if page.getcode() is not 200:
            if verbose is True:
                print('Tried to download data from %s and '
                      'got http response code %s' % (url, str(page.getcode())))
            return False
        else:
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading file: %s. Progress: %.5f%%' %
                                 (output_filename, (float(count*block_size)/(float(total_size)*100.0))))
                sys.stdout.flush()
            print('Downloading the CIFAR 10 dataset from %s to %s' % (url, output_filename))
            downloaded_file, _ = url_req.urlretrieve(url=url, filename=output_filename, reporthook=_progress)
            print()
            if verbose is True:
                print('Moving the file to %s' %output_filename)
            shutil.move(downloaded_file, output_filename)
        return True
    except:
        raise IOError('Error downloading data from %s to %s', url, output_filename)


def extract(file_name, dest_dir='', verbose=False):
    """
    Extract a compressed file
    :param file_name: File to uncompressed
    :param dest_dir: Directory in which the extracted files have to be placed
    :param verbose: If log information is to be printed onto screen
    :return:
    """
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
    """
    Untar a file
    :param source_filename: File to be untarred
    :param destination_dir: Destination directory to which the files have to be extracted
    :param verbose: If log information is to be printed onto screen
    :return:
    """
    source_filename = os.path.realpath(source_filename)
    destination_dir = os.path.realpath(destination_dir)
    if verbose is True:
        print('Unzipping tarball data from %s to %s', source_filename, destination_dir)
    try:
        with tarfile.open(source_filename) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(tar, destination_dir)
            tar.close()
            return True
    except:
        raise IOError('Error unzipping tarball data from %s to %s', source_filename, destination_dir)


def unpickle(file, verbose=False):
    """
    Unpickle a pickled file using cPickle
    :param file: File to be unpickled
    :param verbose: If log information is to be printed onto screen
    :return:
    """
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding ='bytes')
    fo.close()
    return dict


if __name__ == '__main__':
    print('Running file_utils.py')