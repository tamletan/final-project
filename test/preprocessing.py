import pandas as pd
from collections import Counter
from tflearn.data_utils import pad_sequences
import random
import numpy as np
import h5py
import pickle
print("import package successful...")

# read source file as csv
base_path='data/sample/'
train_data_x=pd.read_csv(base_path+'question_train_set3.txt',sep='\t', encoding="utf-8")
train_data_y=pd.read_csv(base_path+'question_topic_train_set3.txt',sep='\t', encoding="utf-8")
valid_data_x=pd.read_csv(base_path+'question_eval_set3.txt', sep='\t',encoding="utf-8")

train_data_x=train_data_x.fillna('')
train_data_y=train_data_y.fillna('')
valid_data_x=valid_data_x.fillna('')
print("train_data_x:",train_data_x.shape)
print("train_data_y:",train_data_y.shape)
print("valid_data_x:",valid_data_x.shape)

# understand your data: that's take a look of data
print(train_data_x.head())

#################################################################################################################

# compute average length of title_char, title_word, desc_char, desc_word

dict_length_columns={'title_char':0,'title_word':0,'desc_char':0,'desc_word':0}
num_examples=len(train_data_x)
train_data_x_small=train_data_x.sample(frac=0.01)
for index, row in train_data_x_small.iterrows():
    title_char_length=len(row['title_char'].split(","))
    title_word_length=len(row['title_word'].split(","))
    desc_char_length=len(row['desc_char'].split(","))
    desc_word_length=len(row['desc_word'].split(","))
    dict_length_columns['title_char']=dict_length_columns['title_char']+title_char_length
    dict_length_columns['title_word']=dict_length_columns['title_word']+title_word_length
    dict_length_columns['desc_char']=dict_length_columns['desc_char']+desc_char_length
    dict_length_columns['desc_word']=dict_length_columns['desc_word']+desc_word_length
dict_length_columns={k:float(v)/float(num_examples*0.01) for k,v in dict_length_columns.items()}
print("dict_length_columns:",dict_length_columns)

print(train_data_y.head())

#################################################################################################################

# average labels for a input
train_data_y_small=train_data_y.sample(frac=0.01)
num_examples=len(train_data_y_small)
num_labels=0
for index, row in train_data_y_small.iterrows():
    topic_ids=row['topic_ids']
    topic_id_list=topic_ids.split(",")
    num_labels+=len(topic_id_list)
average_num_labels=float(num_labels)/float(num_examples)
print("average_num_labels:",average_num_labels)

print(valid_data_x.head())
print(train_data_y.head())

#################################################################################################################

# create vocabulary_dict, label_dict, generate training/validation data, and save to some place 
    
 # create vocabulary of charactor token by read word_embedding.txt 
word_embedding_object=open(base_path+'unused_current/char_embedding.txt')
lines_wv=word_embedding_object.readlines()
word_embedding_object.close()
char_list=[]
char_list.extend(['PAD','UNK','CLS','SEP','unused1','unused2','unused3','unused4','unused5'])
PAD_ID=0
UNK_ID=1
for i, line in enumerate(lines_wv):
    if i==0: continue
    char_embedding_list=line.split(" ")
    char_token=char_embedding_list[0]
    char_list.append(char_token)    
    
# write to vocab.txt under data/ieee_zhihu_cup
vocab_path=base_path+'vocab.txt'
vocab_char_object=open(vocab_path,'w')

word2index={}
for i, char in enumerate(char_list):
    if i<10:print(i,char)
    word2index[char]=i
    vocab_char_object.write(char+"\n")
vocab_char_object.close()
print("vocabulary of char generated....")

#################################################################################################################

# generate labels list, and save to file system 
c_labels=Counter()
train_data_y_small=train_data_y[0:100000]#.sample(frac=0.1)
for index, row in train_data_y_small.iterrows():
    topic_ids=row['topic_ids']
    topic_list=topic_ids.split(',')
    c_labels.update(topic_list)

label_list=c_labels.most_common()
label2index={}
label_target_object=open(base_path+'label_set.txt','w')
for i, label_freq in enumerate(label_list):
    label,freq=label_freq
    label2index[label]=i
    label_target_object.write(label+"\n")
    if i<20: print(label,freq)
label_target_object.close()
print("generate label dict successful...")

#################################################################################################################

def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

label_list=[0,1,2,10]
label_size=20
label_list_sparse=transform_multilabel_as_multihot(label_list,label_size)
print("label_list_sparse:",label_list_sparse)

#################################################################################################################

def get_X_Y(train_data_x,train_data_y,label_size, test_mode=False):
    """
    get X and Y given input and labels
    input:
    train_data_x:
    train_data_y:
    label_size: number of total unique labels(e.g. 1999 in this task)
    output:
    X,Y
    """
    X=[]
    Y=[]
    if test_mode:
        train_data_x_tiny_test=train_data_x[0:1000] # todo todo todo todo todo todo todo todo todo todo todo todo 
        train_data_y_tiny_test=train_data_y[0:1000] # todo todo todo todo todo todo todo todo todo todo todo todo 
    else:
        train_data_x_tiny_test=train_data_x
        train_data_y_tiny_test=train_data_y

    for index, row in train_data_x_tiny_test.iterrows():
        if index==0: continue
        # get character of title and dssc
        title_char=row['title_char']
        desc_char=row['desc_char']
        # split into list
        title_char_list=title_char.split(',')
        desc_char_list=desc_char.split(",")
        # transform to indices
        title_char_id_list=[vocabulary_word2index.get(x,UNK_ID) for x in title_char_list if x.strip()]
        desc_char_id_list=[vocabulary_word2index.get(x,UNK_ID) for x in desc_char_list if x.strip()]
        # merge title and desc: in the middle is special token 'SEP'
        title_char_id_list.append(vocabulary_word2index['SEP'])
        title_char_id_list.extend(desc_char_id_list)
        X.append(title_char_id_list)
        if index<3: print(index,title_char_id_list)
        if index%100000==0: print(index,title_char_id_list)

    for index, row in train_data_y_tiny_test.iterrows():
        if index==0: continue
        topic_ids=row['topic_ids']
        topic_id_list=topic_ids.split(",")
        label_list_dense=[label2index[l] for l in topic_id_list if l.strip()]
        label_list_sparse=transform_multilabel_as_multihot(label_list_dense,label_size)
        Y.append(label_list_sparse)
        if index%100000==0: print(index,";label_list_dense:",label_list_dense)

    return X,Y

print(vocabulary_word2index['SEP'])

def save_data(cache_file_h5py,cache_file_pickle,word2index,label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y):
    # train/valid/test data using h5py
    f = h5py.File(cache_file_h5py, 'w')
    f['train_X'] = train_X
    f['train_Y'] = train_Y
    f['vaild_X'] = vaild_X
    f['valid_Y'] = valid_Y
    f['test_X'] = test_X
    f['test_Y'] = test_Y
    f.close()
    # save word2index, label2index
    with open(cache_file_pickle, 'ab') as target_file:
        pickle.dump((word2index,label2index), target_file)

#################################################################################################################

# generate training/validation/test data using source file and vocabulary/label set.
#  get X,Y---> shuffle and split data----> save to file system.
test_mode=False
label_size=len(label2index)
cache_path_h5py=base_path+'data.h5'
cache_path_pickle=base_path+'vocab_label.pik'
max_sentence_length=200

# step 1: get (X,y) 
X,Y=get_X_Y(train_data_x,train_data_y,label_size,test_mode=test_mode)

# pad and truncate to a max_sequence_length
X = pad_sequences(X, maxlen=max_sentence_length, value=0.)  # padding to max length

# step 2. shuffle, split,
xy=list(zip(X,Y))
random.Random(10000).shuffle(xy)
X,Y=zip(*xy)
X=np.array(X); Y=np.array(Y)
num_examples=len(X)
num_valid=20000
num_valid=20000
num_train=num_examples-(num_valid+num_valid)
train_X, train_Y=X[0:num_train], Y[0:num_train]
vaild_X, valid_Y=X[num_train:num_train+num_valid], Y[num_train:num_train+num_valid]
test_X, test_Y=X[num_train+num_valid:], Y[num_train+num_valid:]
print("num_examples:",num_examples,";X.shape:",X.shape,";Y.shape:",Y.shape)
print("train_X:",train_X.shape,";train_Y:",train_Y.shape,";vaild_X.shape:",vaild_X.shape,";valid_Y:",valid_Y.shape,";test_X:",test_X.shape,";test_Y:",test_Y.shape)

# step 3: save to file system
save_data(cache_path_h5py,cache_path_pickle,word2index,label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y)
print("save cache files to file system successfully!")

del X,Y,train_X, train_Y,vaild_X, valid_Y,test_X, test_Y


topic_info_data=pd.read_csv(base_path+'topic_info.txt', sep='\t',encoding="utf-8")