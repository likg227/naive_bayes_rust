# -*- coding: utf-8 -*-

import os
import shutil
import cchardet as chardet
import codecs


def convert_file_to_utf8(filename):
    # !!! does not backup the origin file
    file_content = codecs.open(filename, 'rb').read()
    source_encoding = chardet.detect(file_content)['encoding']
    if source_encoding is None:
        return False
    if source_encoding != 'utf-8' and source_encoding != 'UTF-8-SIG':
        file_content = file_content.decode(source_encoding, 'ignore')  # .encode(source_encoding)
        codecs.open(filename, 'w', encoding='UTF-8').write(file_content)
    return True


def copy_file(filelist, new_file_path):
    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)

    for file in filelist:
        file_name = file.split('/')
        old_file_path = '/Users/likaige/Code/course_design/naive_bayes_rust/data/email_data' + file
        if not convert_file_to_utf8(old_file_path):
            continue
        shutil.copyfile(old_file_path, new_file_path + '/' + file_name[-2] + '_' + file_name[-1])


if __name__ == '__main__':
    norm_file_list = []
    spam_file_list = []
    # 读index文件
    index_file = '/Users/likaige/Code/course_design/naive_bayes_rust/data/email_data/full/index'
    with open(index_file, 'r') as pf:
        for line in pf.readlines():
            content = line.strip().split('..')
            label, path = content
            if label == 'spam ':
                spam_file_list.append(path)
            else:
                norm_file_list.append(path)

    copy_file(norm_file_list, '/Users/likaige/Code/course_design/naive_bayes_rust/data/email_data/norm')
    copy_file(spam_file_list, '/Users/likaige/Code/course_design/naive_bayes_rust/data/email_data/spam')
