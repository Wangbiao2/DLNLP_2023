import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s', level=logging.INFO)
from opencc import OpenCC
opencc = OpenCC('t2s')
import os
import math
import logging
import numpy as np
import jieba
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import time


def stop_punctuation(path):  # 中文字符表
    with open(path, 'r', encoding='UTF-8') as f:
        items = f.read()
        return [l.strip() for l in items]

def read_data(data_dir):
    data_txt = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_fullname = os.path.join(root, file)
            logging.info('Read file: %s' % file_fullname)

            with open(file_fullname, 'r', encoding='ANSI') as f:
                data = f.read()
                ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'  # 替换每本小说中的小说网络来源词
                data = data.replace(ad, '')

                data_txt.append(data)

            f.close()

    return data_txt, files

def get_idx(args):
    return args

def SentencePreprocessing():
    line = ''
    data_txt, filenames = read_data(data_dir='./data')
    len_data_txt = len(data_txt)
    punctuations = stop_punctuation('./CN_stopwords/cn_stopwords.txt')  # 停词

    # 获取整个语料库
    with open('./novel_sentence.txt', 'w', encoding='utf-8') as f:
        p = multiprocessing.Pool(60)  # 多线程
        args = [i for i in range(len_data_txt)]  # 小说编号
        pbar = tqdm(range(len_data_txt))  # 进度条
        for i in p.imap(get_idx, args):
            pbar.update()  # 更新进度
            text = data_txt[i]  # 第i本小说
            for x in text:
                if x in ['\n', '。', '？', '！', '，', '；', '：'] and line != '\n':  # 以部分中文符号为分割换行
                    if line.strip() != '':
                        f.write(line.strip() + '\n')  # 按行存入语料文件
                        line = ''
                elif x not in punctuations:
                    line += x

            pbar.set_description("选取中文金庸小说篇数: %d - %s" % ((i + 1), filenames[i][:-4]))

        p.close()
        p.join()
        f.close()
        pbar.close()

    # 获取每个单独的语料库
    p = multiprocessing.Pool(60)
    args = [i for i in range(len_data_txt)]
    pbar = tqdm(range(len_data_txt))
    for i in p.imap(get_idx, args):
        with open('./novel_sentence_%s.txt' % filenames[i][:-4], 'w', encoding='utf-8') as f:
            text = data_txt[i]
            pbar.update()
            for x in text:
                if x in ['\n', '。', '？', '！', '，', '；', '：'] and line != '\n':  # 以部分中文符号为分割换行
                    if line.strip() != '':
                        f.write(line.strip() + '\n')  # 按行存入语料文件
                        line = ''
                elif x not in punctuations:
                    line += x

            pbar.set_description("选取中文金庸小说篇数: %d - %s" % ((i + 1), filenames[i][:-4]))
        f.close()

    p.close()
    p.join()
    pbar.close()

# 词频统计(没用)
def get_tf_1(words):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return tf_dic.items()

# 一元模型词频统计
def get_unigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1

# 二元模型词频统计
def get_bigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1

# 三元模型词频统计
def get_trigram_tf(tf_dic, words):
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1

# 计算一元模型信息熵
def calculate_unigram_entropy(file_path,words_tf,len_):
    before = time.time()

    begin = time.time()
    words_num = sum([item[1] for item in words_tf.items()])
    logging.info(file_path)
    print("分词个数：{}".format(words_num))
    print('分词种类数：{}'.format((len(words_tf))))
    print("平均词长：{:.4f}".format(len_ / float(words_num)))

    entropy = 0
    for item in words_tf.items():
        entropy += -(item[1] / words_num) * math.log(item[1] / words_num, 2)
    print("基于词的一元模型中文信息熵为：{:.4f} 比特/词".format(entropy))

    end = time.time()
    runtime = round(end - before, 4)
    print("一元模型运行时间：{:.4f} s".format(runtime))
    return entropy


# 计算二元模型信息熵
def calculate_bigram_entropy(file_path,words_tf,bigram_tf):
    begin = time.time()
    bi_words_num = sum([item[1] for item in bigram_tf.items()])
    avg_word_len = sum(len(item[0][i]) for item in bigram_tf.items() for i in range(len(item[0]))) / len(bigram_tf)
    logging.info(file_path)
    print("分词个数：{}".format(bi_words_num))
    print('分词种类数：{}'.format((len(bigram_tf))))
    print("平均词长：{:.4f}".format(avg_word_len))

    entropy = 0
    for bi_item in bigram_tf.items():
        jp = bi_item[1] / bi_words_num
        cp = bi_item[1] / words_tf[bi_item[0][0]]
        entropy += -jp * math.log(cp, 2)
    print("基于词的二元模型中文信息熵为：{:.4f} 比特/词".format(entropy))

    end = time.time()
    runtime = end - begin
    print("二元模型运行时间：{:.4f} s".format(runtime))
    return entropy

# 计算三元模型信息熵
def calculate_trigram_entropy(file_path,bigram_tf,trigram_tf):
    begin = time.time()
    tri_words_num = sum([item[1] for item in trigram_tf.items()])
    avg_word_len = sum(len(item[0][i]) for item in trigram_tf.items() for i in range(len(item[0]))) / len(trigram_tf)
    logging.info(file_path)
    print("分词个数：{}".format(tri_words_num))
    print('分词种类数：{}'.format((len(trigram_tf))))
    print("平均词长：{:.4f}".format(avg_word_len))

    entropy = 0
    for tri_item in trigram_tf.items():
        jp = tri_item[1] / tri_words_num
        cp = tri_item[1] / bigram_tf[tri_item[0][0]]
        entropy += -jp * math.log(cp, 2)
    print("基于词的三元模型中文信息熵为：{:.4f} 比特/词".format(entropy))

    end = time.time()
    runtime = end-begin
    print("三元模型运行时间：{:.4f} s".format(runtime))
    return entropy

def get_split_words(file_path,flag):
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = []
        split_words = []
        count = 0
        for line in f:
            if line != '\n':
                corpus.append(line.strip())
                count += len(line.strip())

        corpus = ''.join(corpus)
        if flag is False:
            split_words = list(jieba.cut(corpus))  # 利用jieba分词
        elif flag is True:
            split_words = [x for x in corpus]
    return split_words,len(corpus)

def Calculate_total_entropy(file_path,flag):  # 按词/词，计算全部的信息熵
    split_words,len_ = get_split_words(file_path,flag)
    words_tf = {}
    bigram_tf = {}
    trigram_tf = {}

    get_unigram_tf(words_tf, split_words)
    get_bigram_tf(bigram_tf, split_words)
    get_trigram_tf(trigram_tf, split_words)

    print("------------------------------------------------------------------------")
    # unigram
    data = []
    item = calculate_unigram_entropy(file_path,words_tf, len_)
    data.append(item)
    # biggram
    item = calculate_bigram_entropy(file_path,words_tf, bigram_tf)
    data.append(item)
    # trigram
    item = calculate_trigram_entropy(file_path,bigram_tf, trigram_tf)
    data.append(item)
    # 平均信息熵
    entropy = data
    logging.info(file_path+'----Average entropy: %.4f' % (sum(entropy) / len(entropy)))
    print("------------------------------------------------------------------------")


def draw_results(data,novel_name,color,title,path):  # 画柱状图
    length = len(data)
    x = np.arange(length)
    plt.figure(figsize=(12.96, 7.2))
    width = 0.6  # 单个柱状图的宽度
    x1 = x + width / 2  # 第一组数据柱状图横坐标起始位置
    plt.title(title, fontsize=18)  # 柱状图标题
    plt.ylabel("Bit per word", fontsize=15)  # 纵坐标label
    plt.bar(x1, data, width=width, color=color)
    plt.xticks(x, novel_name, rotation=25, fontsize=12)
    for a, b in zip(x1, data):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=12)
    # plt.legend()  # 给出图例
    # plt.show()
    plt.savefig(path)
