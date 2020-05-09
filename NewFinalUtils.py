import json
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler,SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

import collections
# 自己写的一个trange函数, tqdm总出问题
def trange(start_step, max_step, desc):
    last_time = time.time()
    for i in range(start_step + 1, max_step + 1):
        new_time = time.time()
        time_msg = '%.2f' % ((new_time - last_time) * (max_step - i))
        last_time = new_time
        print('\r' + desc + ' :' + str(i) + '/' +
              str(max_step) + ' 还需' + time_msg + '秒', flush=True)
        yield i


# 自己写的tqdm函数， 原因同上
def tqdm(data, desc):
    last_time = time.time()
    max_step = len(data)
    start_step = 1
    for i in data:
        new_time = time.time()
        time_msg = '%.2f' % ((new_time - last_time) * (max_step - start_step))
        last_time = new_time
        print('\r' + desc + ' :' + str(start_step) + '/' +
              str(max_step) + ' 还需' + time_msg + '秒', end='', flush=True)
        start_step += 1
        yield i


# 读取train.json或dev.json时使用
def read_data_file(PATH):
    with open(PATH) as reader:
        return json.load(reader)['data'][0]['paragraphs']


# 读取nbest预测结果时使用
def read_nbest_file(PATH):
    with open(PATH) as reader:
        return json.load(reader)


# 预测答案的包装类
class Answer:
    def __init__(self,
                 text,
                 start_logit,
                 end_logit,
                 probability):
        self.text = text
        self.start_logit = start_logit
        self.end_logit = end_logit
        self.probability = probability


# 真实答案的包装类
class TrueAnswer:
    def __init__(self, text, answer_start):
        self.text = text
        self.answer_start = answer_start


# 数据样本， 包含了问题， 文章， 预测答案和真实答案
class AnswerExample:
    def __init__(self, question_id, context, question, true_answer, answers):
        self.question_id = question_id
        self.context = context
        self.question = question
        self.true_answer = true_answer
        self.answers = answers


# 辅助工具，用于包装数据集
class BaiduMRCDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


# 提供参数的工具类
class ARG:
    def __init__(self,
                 per_gpu_train_batch_size=4,
                 n_gpu=1,
                 eval_batch_size=4,
                 max_steps=-1,
                 weight_decay=0.0,
                 num_train_epochs=3,
                 learning_rate=3e-5,
                 adam_epsilon=1e-8,
                 warmup_steps=0,
                 warmup_proportion=0.1,
                 local_rank=-1,
                 gradient_accumulation_steps=1,
                 device='cpu',
                 max_grad_norm=1.0,
                 save_dir='./tmp',
                 topk=2,
                 model_name='voidful/albert_chinese_base',
                 tokenizer_name='voidful/albert_chinese_base',
                 max_length=256):
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.n_gpu = n_gpu
        self.eval_batch_size = eval_batch_size
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.warmup_proportion = warmup_proportion
        self.local_rank = local_rank
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.topk = topk
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length


# 从train中生成AnswerExample数组，注意这里的true_answer只有一个，且略过了给出的预测答案不足的情况
def build_all_train_examples(examples, nbest_data, TOPK):
    all_examples = []
    for e in examples:
        question_id = e['qas'][0]['id']
        context = e['context']
        question = e['qas'][0]['question']
        true_answer = TrueAnswer(e['qas'][0]['answers'][0]['text'],
                                 e['qas'][0]['answers'][0]['answer_start'])
        answers = []
        all_answers = nbest_data[question_id]
        if len(all_answers) < TOPK:
            continue
        else:
            for i in range(0, TOPK):
                answer_e = Answer(all_answers[i]['text'],
                                  all_answers[i]['start_logit'],
                                  all_answers[i]['end_logit'],
                                  all_answers[i]['probability'])
                answers.append(answer_e)
            all_examples.append(AnswerExample(
                question_id, context, question, true_answer, answers))
    return all_examples


# 从dev中生成AnswerExample数组，这里的true_answer可以有多个
def build_all_dev_examples(examples, nbest_data, TOPK):
    all_examples = []
    for e in examples:
        try:
            question_id = e['qas'][0]['id']
        except:
            continue
        context = e['context']
        question = e['qas'][0]['question']
        true_answers = []
        for obj in e['qas'][0]['answers']:
            true_answers.append(obj['text'])
        answers = []
        all_answers = nbest_data[question_id]
        best_answer_e = ""
        for i in range(len(all_answers)):
            if len(all_answers[i]['text'])>30:
                continue
            
            try:
                answer_e = Answer(all_answers[i]['text'],
                                  all_answers[i]['start_logit'],
                                  all_answers[i]['end_logit'],
                                  all_answers[i]['probability'])
                answers.append(answer_e)
            except Exception:
                print(Exception)
            if len(answers) == TOPK:
                break
        
        if len(answers) < TOPK:
            answers.extend([answers[0]]*(TOPK - len(answers)))

        all_examples.append(AnswerExample(question_id,
                                          context,
                                          question,
                                          true_answers,
                                          answers))
    return all_examples


# 辅助工具，在训练前摇一摇标签
def shuffle_examples(examples):
    import random
    for e in examples:
        random.shuffle(e.answers)
    return examples


# 终极版的训练器， 再也不需要考虑不同模型的问题了， 输入为 参数:ARG, 训练数据集:Dataset， 模型
def final_train_function(args, train_dataset, model):
    """ Train the model """
    model.to(args.device)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
        )

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )

    # 清空模型缓存目录
    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            keys = batch.keys()
            inputs = {}
            for key in keys:
                inputs[key] = batch[key].to(args.device)

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        # 在这个地方打印出本轮平均loss
        print('\n' + str(tr_loss / global_step))

        # 自己写的保存模型的代码
        try:
            PATH = args.save_dir + '/EP' + str(_)
            os.mkdir(PATH)
            model.save_pretrained(PATH)
            print('当前模型保存在：' + PATH)
        except Exception:
            print(Exception)

    return global_step, tr_loss / global_step

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def final_dev_function(args, eval_dataset,eval_examples, model, use_logit=False):
    
    model.to(args.device)    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    resuts = collections.OrderedDict()

    k = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        keys = batch.keys()
            
        inputs = {}
        for key in keys:
            if key != "index" and key != "label":
                inputs[key] = batch[key].to(args.device)
       
        outputs = model(**inputs)


        for i,index in enumerate(batch["index"]):
           

            output = [output[i] for output in outputs]
            
            example = eval_examples[index] 
    
            logit = output[0]

            if not use_logit:
                resuts[example.question_id] = example.answers[torch.argmax(logit).detach().cpu()].text
            else:
                probs = softmax(np.array([answer.start_logit + answer.end_logit for answer in example.answers]))
                probs = probs * logit.detach().cpu().numpy()
                resuts[example.question_id] = example.answers[np.argmax(probs)].text

    return resuts
