#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/9/11


import numpy
from tqdm import tqdm
from sklearn import metrics
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from utils import *


class data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, pattern="", is_pre=True, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.is_pre = is_pre

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, text in self.sample(random):
            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=self.pattern, second_text=text, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text, second_text=self.pattern, maxlen=maxlen)
            source_ids, target_ids = token_ids[:], token_ids[:]
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids, = [], []

def predict(data_generator_list, data):
    print("\n*******************Start to Zero-Shot predict*******************", flush=True)
    patterns_logits = [[] for _ in patterns]
    samples_logits = [[] for _ in data]
    for i in range(len(data_generator_list)):
        print("\nPattern{}".format(i), flush=True)
        data_generator = data_generator_list[i]
        counter = 0
        for (x, _) in tqdm(data_generator):
            outputs = model.predict(x[:2])
            for out in outputs:
                logit_pos = out[0].T
                patterns_logits[i].append(logit_pos)
                samples_logits[counter].append(logit_pos)
                counter += 1
    preds = []
    for i in range(len(patterns_logits[0])):
        pred = numpy.argmax([logits[i] for logits in patterns_logits])
        preds.append(int(pred))
    return preds, samples_logits

if __name__ == "__main__":

    # Load the hyper-parameters-----------------------------------------------------------
    maxlen = 128  # The max length 128 is used in our paper
    batch_size = 40  # Will not influence the results

    # Choose a model----------------------------------------------------------------------
    # Recommend to use 'uer-mixed-bert-base'
    # model_names = ['google-bert', 'google-bert-small', 'google-bert-zh',
    #                'hfl-bert-wwm', 'hfl-bert-wwm-ext',
    #                'uer-mixed-bert-tiny', 'uer-mixed-bert-small',
    #                'uer-mixed-bert-base', 'uer-mixed-bert-large']
    model_name = 'google-bert-zh'

    # Choose a dataset----------------------------------------------------------------------
    # dataset_names = ['eprstmt', 'tnews', 'csldcp', 'iflytek']
    # dataset_name = 'eprstmt'

    # Load model and dataset class
    bert_model = Model(model_name=model_name)

    # Create a template --------------------------------------------------------------------
    #label_names = ['娱乐', '体育', '音乐', '电竞', '经济', '教育']
    #patterns = ["这是一篇{}新闻".format(label) for label in label_names]
    label_names = ['有精神病','沒病','有憂鬱症','有亞斯柏格症','有焦慮症']
    patterns = ["這是一句{}的句子。重鬱症、憂鬱症、亞斯柏格症、焦慮症都是精神病。".format(label) for label in label_names]
    # Prefix or Suffix-------------------------------------------------------------------
    is_pre = True

    # Load the demo set--------------------------------------------------------------------
    # demo_data_zh = ['梅西超越贝利成为南美射手王',
    #              '贾斯汀比伯发布新单曲',
    #              '比心APP被下架并永久关闭陪玩功能',
    #              '徐莉佳的伦敦奥运金牌氧化了',
    #              '10元芯片卖400元!芯片经销商被罚',
    #              '北京首批校外培训机构白名单公布']
    

    demo_data_zh = ['結果有亞斯柏格症的人竟然是我', # ==== pos =====
                 '自從憂鬱症過後不知道是疾病還是藥物的因素',
                 '憂鬱症過後不知道是疾病還是藥物的因素',
                 '憂鬱症治療超過一年',
                 '精神科醫師判定我為無精神症的重鬱症',
                 '我覺得我有社交焦慮症',
                 '醫生說我只是輕微憂鬱症',
                 '我被判定這種病很久了之前還因為嚴重解離現象住院過',
                 '最後醫生開藥時問我讀什麼的',  # ===== neg =====
                 '上一堆奇怪的心靈課程還硬拉小孩要學',
                 '可是再唸下去我不開心啊焦慮症會發作',
                 '竟然被說是無法讀懂肢體語言和表情',
                 '我也希望我能爬得起床啊我鬧鐘已經設了十幾個',
                 '知道爸爸壽命竟然不是從算命師說出來',
                 '就連手機看詩詞都要看個三次才能看完',
                 '有六個月每天晚上都躲在棉被裡哭'
                 ]


    demo_data = demo_data_zh
    demo_generator_list = []
    for p in patterns:
        demo_generator_list.append(data_generator(pattern=p, is_pre=is_pre, data=demo_data, batch_size=batch_size))

    # Build BERT model---------------------------------------------------------------------
    tokenizer = Tokenizer('.' + bert_model.dict_path, do_lower_case=True)
    # Load BERET model with NSP head
    model = build_transformer_model(
        config_path='.' + bert_model.config_path, checkpoint_path='.' + bert_model.checkpoint_path, with_nsp=True,
    )

    # Zero-Shot predict and evaluate-------------------------------------------------------
    preds, samples_logits = predict(demo_generator_list, demo_data)
    for i, (p, d) in enumerate(zip(preds, demo_data)):
        pred_label = label_names[p]
        print("Sample {}:".format(i))
        print("Original Text: {}".format(d))
        print("Predict label: {}".format(pred_label))
        print("Logits: {}".format(samples_logits[i]))
        print()
