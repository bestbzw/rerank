import torch
from transformers import BertTokenizer

from MyNewModel2 import (BertForAnswerSelectionWithConcat)
from NewFinalUtils import (ARG, BaiduMRCDataset, build_all_train_examples,build_all_dev_examples, final_train_function, final_dev_function,read_data_file,
                           read_nbest_file, shuffle_examples, tqdm)
import json
import os
import logging
logger = logging.getLogger(__name__)

# 输出为 Q+C+A1+A2+..., 额外特征[A1的, A2的, ...]， 标签
def convert_feature_1(example, SEP_TOKEN, TOPK, USE_CONTEXT=True, GET_LABEL=False):
    QUESTION = example.question
    CONTEXT = example.context
    ANSWERS = example.answers
    START_INDEX = 9999
    END_INDEX = 0
    EXTENED_FEATURE = []
    if GET_LABEL:
        TRUE_ANSWER = example.true_answer.text
    ANSWERS_TEXT = ''
    FEATURE_TEXT = QUESTION


    for i in range(TOPK):
        answer = ANSWERS[i]
        answer_text = answer.text
        answer_start_logit = answer.start_logit
        answer_end_logit = answer.end_logit
        answer_probability = answer.probability
        answer_start_index = CONTEXT.find(answer_text)
        answer_end_index = answer_start_index + len(answer_text)
        EXTENED_FEATURE += [answer_start_logit,
                            answer_end_logit,
                            answer_probability,
                            answer_start_index / len(CONTEXT),
                            answer_end_index / len(CONTEXT)]

        if i== 0 :
            ANSWERS_TEXT += answer_text
        else:
            ANSWERS_TEXT += SEP_TOKEN + answer_text
        
        if answer_start_index < START_INDEX:
            START_INDEX = answer_start_index
        if answer_end_index > END_INDEX:
            END_INDEX = answer_end_index

    START_INDEX = START_INDEX - 50 if START_INDEX - 50 >= 0 else 0
    END_INDEX = END_INDEX + 50 if END_INDEX + \
                                  50 < len(CONTEXT) else len(CONTEXT)
    if USE_CONTEXT:
        FEATURE_TEXT += SEP_TOKEN + CONTEXT[START_INDEX:END_INDEX]
    
    #FEATURE_TEXT += ANSWERS_TEXT

    if GET_LABEL:
        LABEL = -1
        for i in range(TOPK):
            answer = ANSWERS[i]
            answer_text = answer.text
            if answer_text == TRUE_ANSWER:
                LABEL = i
                break
        return FEATURE_TEXT, ANSWERS_TEXT, EXTENED_FEATURE, LABEL
    else:
        return FEATURE_TEXT, ANSWERS_TEXT, EXTENED_FEATURE

def GetAnswerIndex(input_ids,tokenizer,USE_CONTEXT):
    sep_num = 0
    flag=False
    answer_index = []
    for i, _id in enumerate(input_ids):
        if flag:
            answer_index += [i]
            flag=False
        
        if _id == tokenizer.sep_token_id:
            sep_num += 1

            if (sep_num >= 1 and not USE_CONTEXT) or  (sep_num >= 2 and USE_CONTEXT):
                flag=True
    return answer_index[:-1]

class ConcatAnswerSelectMethod:
    def __init__(self, args: ARG):
        self.train_examples = None
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        self.model = BertForAnswerSelectionWithConcat.from_pretrained(
            args.model_name, num_labels=args.topk)
        # self.convert_feature = convert_feature_1
        self.datas = None

    def read_all_files(self, NBEST_FILE_PATH, DATA_FILE_PATH,evaluate=False):
        if not evaluate:
            self.train_examples = build_all_train_examples(read_data_file(DATA_FILE_PATH),
                                                       read_nbest_file(
                                                           NBEST_FILE_PATH),
                                                       self.args.topk)
            self.train_examples = shuffle_examples(self.train_examples)

        else:
            self.dev_examples = build_all_dev_examples(read_data_file(DATA_FILE_PATH),
                                                        read_nbest_file(
                                                        NBEST_FILE_PATH),
                                                        self.args.topk)

    def build_train_data(self, USE_CONTEXT=False):
        datas = []
        for e in tqdm(self.train_examples, desc='生成训练数据'):
            feature_text,answer_text, ext_feature, label = convert_feature_1(example=e,
                                                            SEP_TOKEN='[SEP]',
                                                            TOPK=self.args.topk,
                                                            USE_CONTEXT=USE_CONTEXT,
                                                            GET_LABEL=True)

            encoded_feature = self.tokenizer.encode_plus(text = feature_text,
                                                         text_pair = answer_text,
                                                         add_special_tokens=True,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_length)

            #BZW: 增加answer_index:
            answer_index = GetAnswerIndex(encoded_feature['input_ids'],self.tokenizer,USE_CONTEXT)
            
            #BZW: 增加token_type_ids用于分割question & answer
            result = {'input_ids': torch.LongTensor(encoded_feature['input_ids']),
                      'attention_mask': torch.LongTensor(encoded_feature['attention_mask']),
                      'additional_feature': torch.FloatTensor(ext_feature),
                      'token_type_ids': torch.LongTensor(encoded_feature['token_type_ids']),
                      'labels': label,
                      'answer_index': torch.LongTensor(answer_index)}

            # 没有正确答案的丢了
            if label != -1:
                datas.append(result)

        self.datas = datas

    def build_dev_data(self,USE_CONTEXT=False):
        datas = []
        for i,e in tqdm(list(enumerate(self.dev_examples)), desc='生成测试数据'):


            feature_text,answer_text, ext_feature = convert_feature_1(example=e,
                                                            SEP_TOKEN='[SEP]',
                                                            TOPK=self.args.topk,
                                                            USE_CONTEXT=USE_CONTEXT,
                                                            GET_LABEL=False)

            encoded_feature = self.tokenizer.encode_plus(text = feature_text,
                                                         text_pair = answer_text,
                                                         add_special_tokens=True,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_length)

            #BZW: 增加answer_index:
            answer_index = GetAnswerIndex(encoded_feature['input_ids'],self.tokenizer,USE_CONTEXT)
            
            #BZW: 增加token_type_ids用于分割question & answer
            result = {'input_ids': torch.LongTensor(encoded_feature['input_ids']),
                      'attention_mask': torch.LongTensor(encoded_feature['attention_mask']),
                      'additional_feature': torch.FloatTensor(ext_feature),
                      'token_type_ids': torch.LongTensor(encoded_feature['token_type_ids']),
                      'answer_index': torch.LongTensor(answer_index),
                      'index': torch.LongTensor([i])}
            
            datas.append(result)
        self.datas = datas     

    def fit(self):
        train_data = BaiduMRCDataset(self.datas)
        final_train_function(self.args, train_data, self.model)


    def fit_eval(self,use_logit=False):
        dev_data = BaiduMRCDataset(self.datas)
        results = final_dev_function(self.args,dev_data,self.dev_examples,self.model,use_logit=use_logit)

        json.dump(results,open(os.path.join(self.args.save_dir,"result.json"),"w"),ensure_ascii=False,indent=4)
