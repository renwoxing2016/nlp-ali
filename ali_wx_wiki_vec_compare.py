# -*- coding: utf-8 -*-
"""
Created on 2018

@author: 
"""

# ####### 2个text 比较相似度 ################
# #输入原始数据 
# #
# # 此处理是将语句1 语句2 进行比较给出相似度
# #
# #输出是....
# #使用方法：python compare.py ./input_test.txt ./temp/
# #
# #
import sys

import logging
import gensim
import jieba

#from gensim.corpora import WikiCorps
from gensim.models import Word2Vec

vec_modelPath = './models/'
vec_filePath = './'

# 使用微信语料训练时的参数
vec_size = 256
vec_window = 10  
vec_minCount = 64
vec_workers = 25


# #######################
def compare_text_on_vec_model(textfile1,textfile2):
    #获取日志信息  
    logging.basicConfig(format = '%(asctime)s:%(levelname)s:%(message)s',level = logging.INFO) 
    
    # #导入模型
    modelfile = 'ali.text.vec.model'
    #modelfile = 'wiki_wx_ali.vec.model'
    model = Word2Vec.load(vec_modelPath + modelfile)
    
    arraydata = []
    # 计算两个集合之间的余弦似度
    jieba.load_userdict('user_dict.txt')
    
    txtFile = open(textfile1,'r',encoding='utf-8')
    for line in txtFile.readlines():
        text_temp = line.split('\t')
        
        number = text_temp[0]
        text1 = text_temp[1]
        text2 = text_temp[2]
        print('number：',number)
        print('text1：',text1)
        print('text2：',text2)
        
        list1 = list(jieba.cut(text1))         # 1.分词
        list2 = list(jieba.cut(text2))         # 1.分词
        print(list1)
        print(list2)
        
        try:
            list_sim1 = model.n_similarity(list1,list2)
            print('text1,text2 相似度：',list_sim1)
        except:
            print('it has except...')
            list_sim1 = 0
        
        result = 0
        if(list_sim1 > 0.5):
            result = 1
        
        arraydata.append([number,result])
    txtFile.close()
    
    outfile = open(textfile2, 'w', encoding='utf-8')
    for item in arraydata:
        lineseg = ''
        lineseg = lineseg + str(item[0]) + '\t'
        lineseg = lineseg + str(item[1])
        #print(lineseg)
        outfile.write(lineseg)
        outfile.write('\n')
    outfile.close()
    
# # using demo
# # python compare.py ./input_test.txt ./temp/
# #
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python compare.py inputfile')
        sys.exit()

    #input_file = sys.argv[1]
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    print('argv[1]:',input1)  
    print('argv[2]:',input2)
    
    input2 = input2 + 'out_result.txt'
    print('output file:',input2)
    
    print('start compare text data')
    
    # 对2个输入的语句 进行比较相似度
    compare_text_on_vec_model(input1,input2)
    
    print('compare end')
    
