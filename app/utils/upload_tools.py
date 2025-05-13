__author__ = 'cq'

import random


def img_allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#获取文件后缀
def get_filetype(filename):
    return '.' + filename.rsplit('.', 1)[1]

#生成随机的字符串文件名
def random_name():
    return ''.join(random.sample('1234567890qazxswedcvfrtgbnhyujmkiolp', 10))