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
    my_file = [ ['./novel_sentence.txt'],
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
    data_1_gram_word = [14.3238,12.4196,12.9941,12.6134,13.2114,13.4500,13.2815,11.1289,13.0251,12.8229,13.0726,10.0099,12.3949,12.1144,12.8401,10.9532,13.0667]
    data_2_gram_word = [5.7377,1.3311,3.5262,3.2289,4.0375,4.0881,3.9840,2.4515,3.4084,4.1771,3.9950,1.6360,2.8995,2.4926,3.3968,1.8216,4.3444]
    data_3_gram_word = [0.6441,0.0561,0.2505,0.2798,0.3713,0.3608,0.3092,0.1698,0.2557,0.4229,0.4269,0.2020,0.2038,0.1515,0.2642,0.1142,0.4758]
    data_avg_word =    [6.9019,4.6022,5.5902,5.3740,5.8734,5.9663,5.8582,4.5834,5.5631,5.8076,5.8315,3.9490,5.1661,4.9195,5.5004,4.2964,5.9623]

    novel_name = ['ALL','三十三剑客图','书剑恩仇录','侠客行','倚天屠龙记','天龙八部','射雕英雄传','白马啸西风','碧血剑','神雕侠侣','笑傲江湖','越女剑','连城诀','雪山飞狐','飞狐外传','鸳鸯刀','鹿鼎记']

    data_1_gram_char = [10.0916,10.0504,9.8583, 9.5035,9.7706,9.8886,9.8168,9.1982,9.8010,9.6743,9.6102,8.6164,9.5920, 9.5337,9.6787,9.2049, 9.7429]
    data_2_gram_char = [7.0873,3.9694,5.4701,5.2223,5.8432,5.9968,5.8298,3.8178,5.4771,5.9551,5.8109,2.9117,4.9425,4.5813,5.3701,3.4287,5.9082]
    data_3_gram_char = [3.2491,0.5916,1.7197,1.6920,2.0949,2.1306,2.0308,1.1456,1.6447,2.1144,2.1499,0.8296,1.4334,1.2002,1.7175,0.8411,2.2275]
    data_avg_char =    [6.8094,4.8705,5.6827,5.4726,5.9029,6.0053,5.8925,4.7205,5.6409,5.9146,5.8570,4.1190,5.3226,5.1051,5.5888,4.4916,5.9595]


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