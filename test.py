'''
姓名：王彪
学号：ZY2203114
运行：test.py
功能：计算1元、2元、3元平均信息熵
****************主要函数都位于 ./util/tools.py文件夹*************************
---------------------------------------------------------------------------
环境配置：
conda create -n NLP_homework1 python=3.8
activate NLP_homework1

pip install numpy
pip install math
pip install jieba
pip install matplotlib
pip install logging
pip3 install multiprocessing
pip3 install opencc-python-reimplemented
'''

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
from util.tools import *

if __name__ == '__main__':
    my_file = [['./novel_sentence.txt'],
               ['./novel_sentence_三十三剑客图.txt'],
               ['./novel_sentence_书剑恩仇录.txt'],
               ['./novel_sentence_侠客行.txt'],
               ['./novel_sentence_倚天屠龙记.txt'],
               ['./novel_sentence_天龙八部.txt'],
               ['./novel_sentence_射雕英雄传.txt'],
               ['./novel_sentence_白马啸西风.txt'],
               ['./novel_sentence_碧血剑.txt'],
               ['./novel_sentence_神雕侠侣.txt'],
               ['./novel_sentence_笑傲江湖.txt'],
               ['./novel_sentence_越女剑.txt'],
               ['./novel_sentence_连城诀.txt'],
               ['novel_sentence_雪山飞狐.txt'],
               ['novel_sentence_飞狐外传.txt'],
               ['novel_sentence_鸳鸯刀.txt'],
               ['novel_sentence_鹿鼎记.txt']]
    for i, name in enumerate(my_file):  # 判断每本小说的语料文件是否存在
        if os.path.exists(name[0]) is False:  # 若存在，则不重复处理
            print(name[0] + '----------->' + 'does not exist')
            SentencePreprocessing()  # 若不存在，则预处理
            continue
        else:
            print(name[0]+'----------->'+'exists')

    # 按词/字，计算全部的一元、二元、三元信息熵
    for j, file_name in enumerate(my_file):
        Calculate_total_entropy(file_name[0], flag=False)
    print('*******************************************************************')
    print('********************************按字********************************')
    print('*******************************************************************')
    for j, file_name in enumerate(my_file):
        Calculate_total_entropy(file_name[0], flag=True)
    # 画图,采用预先记录的数据
    data_1_gram_word = [14.3238,12.4196,12.9941,12.6135,13.2114,13.4500,13.2815,11.1289,13.0251,12.8229,13.0726,10.0099,12.3949,12.1144,12.8401,10.9532,13.0667]
    data_2_gram_word = [5.3787,1.4006,3.5019,3.1879,3.9723,3.9611,3.9054,2.3833,3.4451,4.1433,3.9301,1.5652,2.8637,2.5544,3.3577,1.7207,4.1419]
    data_3_gram_word = [0.9819,0.1607,0.5026,0.5604,0.6782,0.6653,0.5102,0.2703,0.4520,0.7912,0.8444,0.5575,0.4076,0.3165,0.5476,0.2418,0.7544]
    data_avg_word = [6.89484,.6603,5.6662,5.4539,5.9540,6.0255,5.8990,4.5942,5.6407,5.9191,5.9490,4.0442,5.2221,4.9951,5.5818,4.3052,5.9877]

    novel_name = ['ALL','三十三剑客图','书剑恩仇录','侠客行','倚天屠龙记','天龙八部','射雕英雄传','白马啸西风','碧血剑','神雕侠侣','笑傲江湖','越女剑','连城诀','雪山飞狐','飞狐外传','鸳鸯刀','鹿鼎记']

    data_1_gram_char = [10.0916,10.0503,9.8583, 9.5035,9.7706,9.8887,9.8168,9.1982,9.8010,9.6743,9.6102,8.6164, 9.5919, 9.5337,9.6787,9.2049, 9.7428]
    data_2_gram_char = [6.1239,3.4441,4.6374,4.4237,5.0116,5.0866,4.9754,3.0427,4.7115,5.1705,4.9143,2.4000,4.1782,3.8860,4.5618,2.7862,4.9523]
    data_3_gram_char = [3.0288,0.7263,1.8821,1.8183,2.1675,2.1783,2.1798,1.3212,1.7905,2.1787,2.1251,0.9969,1.5627,1.3864,1.8642,0.9888,2.2457]
    data_avg_char = [6.4148,4.7402,5.4593,5.2485,5.6499, 5.7179,5.6573,4.5207,5.4343,5.6745,5.5499,4.0044,5.1109,4.9354,5.3682,4.3266,5.6469]


    draw_results(data_1_gram_word, novel_name, 'red', title='1-gram Chinese Information Entropy', path='figs/1-gram-byword.png')
    draw_results(data_2_gram_word, novel_name, 'blue', title='2-gram Chinese Information Entropy', path='figs/2-gram-byword.png')
    draw_results(data_3_gram_word, novel_name, 'green', title='3-gram Chinese Information Entropy', path='figs/3-gram-byword.png')
    draw_results(data_avg_word, novel_name, 'black', title='Chinese Average Information Entropy', path='figs/Average-Entropy-byword .png')

    draw_results(data_1_gram_char, novel_name, 'red', title='1-gram Chinese Information Entropy',
                 path='figs/1-gram-bychar.png')
    draw_results(data_2_gram_char, novel_name, 'blue', title='2-gram Chinese Information Entropy',
                 path='figs/2-gram-bychar.png')
    draw_results(data_3_gram_char, novel_name, 'green', title='3-gram Chinese Information Entropy',
                 path='figs/3-gram-bychar.png')
    draw_results(data_avg_char, novel_name, 'black', title='Chinese Average Information Entropy',
                 path='figs/Average-Entropy-bychar .png')