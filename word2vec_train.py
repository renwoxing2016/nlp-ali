# -*- coding: utf-8 -*-
"""
Created on 2018

@author: 
"""

# #######text data 前提################
# #输入原始数据 
# #
# # 此处理是将中文语料输入 训练，并并保存模型
# #
# #输出是....
# #使用方法：python word2vec_train.py std_zh_wiki_00
# #
# #

import os
import sys
import multiprocessing
import logging

import gensim

#from gensim.corpora import WikiCorps
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


vec_modelPath = './models/'
vec_filePath = './'

vec_size = 400  
vec_window = 5  
vec_minCount = 5  
vec_workers = 2  

# 针对一个语料文件 进行训练
def vec_train_fun(modelpath, input_file, outmodel):
    # inp1为输入语料 outp1 为输出模型 outp2为原始c版本word2vec的vector格式的模型
    inp1 = input_file
    
    outp1 = modelpath + outmodel
    outp2 = modelpath + 'c_' + outmodel
    
    #获取日志信息  
    logging.basicConfig(format = '%(asctime)s:%(leveltime)s:%(message)s',level = logging.INFO) 
    
    # ######### train
    sentences = LineSentence(inp1)
    cpuworkers = multiprocessing.cpu_count()
    
    # size表示神经网络的隐藏层单元数，默认为100
    # window表示
    # min_count小于该数的单词会被剔除，默认值为5,
    model = Word2Vec(sentences, size = vec_size, window = vec_window, min_count = vec_minCount,
                     workers = cpuworkers)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary = False)

# (无初始化模型)针对多个语料文件 进行训练
def vec_increment_train_of_no_model(model_path,incr_filedata):
    # wiki file path
    wikifile = vec_filePath + incr_filedata[0]
    print('corpuse file:',wikifile)
    
    cpuworkers = multiprocessing.cpu_count()
    print('cpu counter:', cpuworkers)

    #获取日志信息  
    #logging.basicConfig(format = '%(asctime)s:%(leveltime)s:%(message)s',level = logging.INFO) 
     
    #fn = open(wikifile, u'r', encoding="utf-8")
    sentences = LineSentence(wikifile)
    
    print('training init model!')
    model = Word2Vec(sentences, size=vec_size, window=vec_window, min_count=vec_minCount,
                     workers=cpuworkers)  
    
    for file in incr_filedata[1:len(incr_filedata)]:
        tempfile = vec_filePath + file
        corpusSingleFile = open(tempfile, u'r', encoding="utf-8")
        
        trainedWordCount = model.train(LineSentence(corpusSingleFile),
                                       total_examples=model.corpus_count,epochs=model.epochs)
        #                               total_examples=model.corpus_count,epochs=model.iter)
        print('update model, update words num is: ', trainedWordCount)
    
    outp1 = model_path + 'zhs_incr.vec.model'
    model.save(outp1) 
    
    return True

# (有初始化模型)针对多个语料文件 进行训练
# 输入中 incr_filedata 是原料的文件名 包括路径+文件名
#       oldmodel 原有模型的文件名 不含路径
#       newmodel 新的模型的文件名 不含路径
def vec_increment_train_fun_on_basemodel(model_path,oldmodel,incr_filedata,newmodel):
    # 加载 原有向量模型
    model = Word2Vec.load(model_path + oldmodel)
    
    for file in incr_filedata[0:len(incr_filedata)]:
        tempfile = file
        corpusSingleFile = open(tempfile, u'r', encoding="utf-8")
        more_sentences = LineSentence(corpusSingleFile)
        
        model.build_vocab(more_sentences, update=True)
        # 进行训练
        model.train(more_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        print('new corpuse is training...')
    
    outp1 = model_path + newmodel
    model.save(outp1)
    
    return True

# #######################

def vec_model_test_func():
    # #导入模型
    model = Word2Vec.load("text8.model")
    
    # 计算两个词的相似度/相关程度  
    y1 = model.similarity(u"不错", u"好")  
    print(u"【不错】和【好】的相似度为：", y1)
    print("--------\n")
      
    # 计算某个词的相关词列表  
    y2 = model.most_similar(u"书", topn=20)  # 20个最相关的  
    print(u"和【书】最相关的词有：\n")
    for item in y2:  
        print(item[0], item[1])
    print("--------\n")
      
    # 寻找对应关系  
    print(u"书-不错，质量-")
    y3 = model.most_similar([u'质量', u'不错'], [u'书'], topn=3)  
    for item in y3:  
        print(item[0], item[1]) 
    print("--------\n")
      
    # 寻找不合群的词  即选出集合中不同类的词语
    y4 = model.doesnt_match(u"书 书籍 教材 很".split())  
    print(u"不合群的词：", y4)
    print("--------\n")
    
    # 计算两个集合之间的余弦似度
    list1 = ['我','走','我','学校']   
    list2 = ['我','去','家']   
    list_sim1 = model.n_similarity(list1,list2)
    print(list_sim1)
    

if __name__ == '__main__':
    #if len(sys.argv) != 2:
    #    print('Usage: python script.py inputfile')
    #    sys.exit()

    #input_file = sys.argv[1]
    # input_file = 'cut_std_zhs_wiki_00'
    input_file = './csvdata/ali_nlp_sim.dat'
    outmodel = 'ali.text.vec.model'
    
    model_path = './models/'
    
    print('start train text data')    
    
    # 针对一个语料文件 进行训练
    #vec_train_fun(model_path,input_file,outmodel)

    # 针对多个语料文件 进行训练
    #incr_corpuses = ['test_00.txt','test_01.txt']
    #vec_increment_train_of_no_model(model_path, incr_corpuses)
    
    # 针对有训练模型的情况 继续增量训练
    #incr_data2 = ["./csvdata/ali_nlp_sim.dat"]
    incr_data2 = ["cut_std_zhs_wiki_00","cut_std_zhs_wiki_01","cut_std_zhs_wiki_02",
                  "cut_std_zhs_wiki_03","cut_std_zhs_wiki_04","cut_std_zhs_wiki_05"]
    #oldmodel = 'word2vec_wx'
    #newmodel = 'word2vec_wx_ali.vec.model'
    
    oldmodel = 'word2vec_wx_ali.vec.model'
    newmodel = 'wiki_wx_ali.vec.model'
    vec_increment_train_fun_on_basemodel(model_path, oldmodel,incr_data2,newmodel)
    
    print('train end text data')
    
