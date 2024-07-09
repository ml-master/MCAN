# -*- coding: utf-8 -*-
import re
import os
from PIL import Image
import numpy as np


data_path = 'E:\code project\Reproduction_of_MCAN-main\GossipCop-LLM-public'
new_train = 'E:\code project\Reproduction_of_MCAN-main\processd_data\Gosip/train.txt'
new_test = 'E:\code project\Reproduction_of_MCAN-main\processd_data\Gosip/test.txt'

image_file_list = [os.path.join(data_path, 'top_img/')]


def read_images(file_list):
    image_list = {}
    #img_num = 0
    for path in file_list:
        for filename in os.listdir(path):
            try:
                img = Image.open(path + filename).convert('RGB')
                img_id = filename.split('.')[0]
                image_list[img_id] = img
                #print('ok')
                #img_num += 1
            except:
                # print(filename)
                pass
    return image_list#, img_num

def select_image(image_num, image_id_list, image_list):
    for i in range(image_num):
        #print('list:{}'.format(image_id_list))
        image_id = image_id_list[i]
        if image_id in image_list:
            #print('Yes, img_id:{}'.format(img_id))
            return image_id
    #f_log.write(line)
    return False


def get_max_len(file):
    # Get the maximal length of sentence in dataset

    f = open(file, 'r', encoding='UTF-8')

    max_post_len = 0

    lines = f.readlines()
    post_num = len(lines)
    for i in range(post_num):
        post_content = list(lines[i].split('|')[1].split())
        tmp_len = len("".join(post_content))
        if tmp_len > max_post_len:
            max_post_len = tmp_len

    f.close()
    return max_post_len


def get_data(dataset, image_list):
    if dataset == 'train':
        data_file = new_train
    else:
        data_file = new_test

    f = open(data_file, 'r', encoding='UTF-8')
    lines = f.readlines()

    data_post_id = []
    data_post_content = []
    data_image = []
    data_label = []

    data_num = len(lines)
    unmatched_num = 0

    for line in lines:
        post_id = line.split('|')[0]
        post_content = line.split('|')[1]
        label = line.split('|')[-1].strip()

        image_id_list = line.split('|')[-2].strip().split(',')
        # print(image_id_list)
        img_num = len(image_id_list)
        image_id = select_image(img_num, image_id_list, image_list)

        if image_id != False:
            image = image_list[image_id]

            data_post_id.append(int(post_id))
            data_post_content.append(post_content)
            data_image.append(image)
            data_label.append(int(label))

        else:
            unmatched_num += 1
            continue

    f.close()

    data_dic = {'post_id': np.array(data_post_id),
                'post_content': data_post_content,
                'image': data_image,
                'label': np.array(data_label)
                }

    # print('post id:', len(data_post_id))
    # print('data_post_content:', len(data_post_content))
    # print('data_image:', len(data_image))
    # print('data_label:', len(data_label))

    return data_dic, data_num - unmatched_num




