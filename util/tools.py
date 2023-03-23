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

# 词频统计
def get_tf_1(words):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return tf_dic.items()

# 词频统计
def get_tf_2(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1

# 三元模型词频统计
def get_trigram_tf(tf_dic, words):
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1


# 二元模型词频统计
def get_bigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1

# 计算二元模型信息熵
def calculate_bigram_entropy(file_path):
    before = time.time()

    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = []
        count = 0
        for line in f:
            if line != '\n':
                corpus.append(line.strip())
                count += len(line.strip())

    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    bigram_tf = {}

    for line in corpus:
        for x in jieba.cut(line):
            split_words.append(x)
            words_len += 1

        get_tf_2(words_tf, split_words)
        get_bigram_tf(bigram_tf, split_words)

        split_words = []
        line_count += 1

    logging.info("语料库字数: %d" % count)
    logging.info("分词个数: %d" % words_len)
    logging.info("平均词长: %.4f" % round(count / words_len, 4))
    logging.info("语料行数: %d" % line_count)

    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    logging.info("二元模型长度: %d" % bigram_len)

    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    entropy = round(sum(entropy), 4)
    logging.info("基于词的二元模型的中文信息熵为（%s）: %.4f 比特/词" % (file_path, entropy))

    after = time.time()
    logging.info("运行时间: %.4f s" % round(after - before, 4))

    runtime = round(after - before, 4)
    return ['bigram', len(corpus), words_len, round(len(corpus) / words_len, 4), entropy, runtime]

# 计算三元模型信息熵
def calculate_trigram_entropy(file_path):
    before = time.time()
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = []
        count = 0
        for line in f:
            if line != '\n':
                corpus.append(line.strip())
                count += len(line.strip())

    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    trigram_tf = {}

    for line in corpus:
        for x in jieba.cut(line):
            split_words.append(x)
            words_len += 1

        get_bigram_tf(words_tf, split_words)
        get_trigram_tf(trigram_tf, split_words)

        split_words = []
        line_count += 1

    logging.info("语料库字数: %d" % count)
    logging.info("分词个数: %d" % words_len)
    logging.info("平均词长: %.4f" % round(count / words_len, 4))
    logging.info("语料行数: %d" % line_count)

    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    logging.info("三元模型长度: %d" % trigram_len)

    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len  # 计算联合概率p(x,y)
        cp_xy = tri_word[1] / words_tf[tri_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
    entropy = round(sum(entropy), 4)
    logging.info("基于词的三元模型的中文信息熵为（%s）: %.4f 比特/词" % (file_path, entropy))  # 0.936

    after = time.time()
    logging.info("运行时间: %.4f" % round(after - before, 4))
    runtime = round(after - before, 4)

    return ['trigram', len(corpus), words_len, round(len(corpus) / words_len, 4), entropy, runtime]


def calculate_unigram_entropy(file_path):
    before = time.time()

    with open(file_path, 'r', encoding='utf-8') as f:

        corpus = []
        count = 0
        for line in f:
            if line != '\n':
                corpus.append(line.strip())
                count += len(line.strip())

        corpus = ''.join(corpus)

        split_words = [x for x in jieba.cut(corpus)]  # 利用jieba分词
        words_len = len(split_words)

        logging.info("语料库字数: %d" % len(corpus))
        logging.info("分词个数: %d" % words_len)
        logging.info("平均词长: %.4f" % round(len(corpus)/words_len, 4))

        words_tf = get_tf_1(split_words)  # 得到词频表

        entropy = [-(uni_word[1]/words_len)*math.log(uni_word[1]/words_len, 2) for uni_word in words_tf]
        entropy = round(sum(entropy), 4)
        logging.info("基于词的一元模型的中文信息熵为（%s）: %.4f 比特/词" % (file_path, entropy))

    after = time.time()
    runtime = round(after - before, 4)
    logging.info("运行时间: %.4f s" % runtime)

    return ['unigram', len(corpus), words_len, round(len(corpus)/words_len, 4),  entropy, runtime]

def Calculate_total_entropy(file_path):  # 计算全部的信息熵
    print("------------------------------------------------------------------------")
    # unigram
    data = []
    item = calculate_unigram_entropy(file_path)
    data.append(item)
    # biggram
    item = calculate_bigram_entropy(file_path)
    data.append(item)
    # trigram
    item = calculate_trigram_entropy(file_path)
    data.append(item)
    # 平均信息熵
    entropy = [item[4] for item in data]
    logging.info(file_path+'----Average entropy: %.4f' % (sum(entropy) / len(entropy)))
    print("------------------------------------------------------------------------")

def draw_results(data,novel_name,color,title,path):  # 画柱状图
    length = len(data)
    x = np.arange(length)
    plt.figure(figsize=(12.96, 7.2))
    width = 0.6  # 单个柱状图的宽度
    x1 = x + width / 2  # 第一组数据柱状图横坐标起始位置
    plt.title(title,fontsize=18)  # 柱状图标题
    plt.ylabel("Bit per word",fontsize=15)  # 纵坐标label
    plt.bar(x1, data, width=width, color=color)
    plt.xticks(x, novel_name, rotation=25, fontsize=12)
    for a, b in zip(x1, data):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=12)
    # plt.legend()  # 给出图例
    # plt.show()
    plt.savefig(path)