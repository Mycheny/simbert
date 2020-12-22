#! -*- coding: utf-8 -*-
# SimBERT训练代码
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

from __future__ import print_function
import json
import os
import random
import sys
import numpy as np
import tensorflow as tf
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import text_segmentate
from bert4keras.snippets import AutoRegressiveDecoder
if __name__ == '__main__': os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# from bert4keras.snippets import uniout

# 基本信息
maxlen = 32
batch_size = 16
steps_per_epoch = 625
epochs = 1000
corpus_path = os.path.join('resources', 'data', 'dataset.json')
best_model_path = os.path.join('resources', 'model', 'best_model.weights')
latest_model_path = os.path.join('resources', 'model', 'latest_model.weights')
if sys.platform == "linux":
    # bert配置
    root = r"resources/bert"
else:
    # bert配置
    root = r"resources/bert"
    # root = r"E:\BaiduNetdiskDownload\chinese_simbert_L-12_H-768_A-12"
config_path = os.path.join(root, 'bert_config.json')
checkpoint_path = os.path.join(root, 'bert_model.ckpt')
dict_path = os.path.join(root, 'vocab.txt')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def get_tensor_values(tensor, bert):
    token_id1, segment_id1 = tokenizer.encode("有现金.住房公积金里有十万，可以买房吗", second_text="用住房公积金贷款买房，同时可以取出现金吗", max_length=maxlen)
    token_id2, segment_id2 = tokenizer.encode("用住房公积金贷款买房，同时可以取出现金吗", second_text="有现金.住房公积金里有十万，可以买房吗", max_length=maxlen)
    token_id3, segment_id3 = tokenizer.encode("晚上好无聊 辛店晚上有好玩的地方吗", second_text="晚上哪里有好玩的，甚是无聊寂寞", max_length=maxlen)
    token_id4, segment_id4 = tokenizer.encode("晚上哪里有好玩的，甚是无聊寂寞", second_text="晚上好无聊 辛店晚上有好玩的地方吗", max_length=maxlen)
    token_ids = [token_id1, token_id2, token_id3, token_id4]
    segment_ids = [segment_id1, segment_id2, segment_id3, segment_id4]
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        out = sess.run(tensor, feed_dict={bert.model.inputs[0]: token_ids, bert.model.inputs[1]: segment_ids})
        return out

def read_corpus():
    """读取语料，每行一个json
    """
    with open(corpus_path, encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    while True:
        for line in lines:
            yield json.loads(line)


def truncate(text):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, d in self.sample(random):
            text, synonyms = d['text'], d['synonyms']
            synonyms = [text] + synonyms
            np.random.shuffle(synonyms)
            text, synonym = synonyms[:2]
            text, synonym = truncate(text), truncate(synonym)
            self.some_samples.append(text)
            if len(self.some_samples) > 1000:
                self.some_samples.pop(0)
            token_ids, segment_ids = tokenizer.encode(
                text, synonym, max_length=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(
                synonym, text, max_length=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """

    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 30  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])  # value=[0, ..., batch-1]
        idxs_1 = idxs[None, :]  # shape=(1, batch)
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]  # shape=(batch, 1)
        labels = K.equal(idxs_1, idxs_2)  # eg: batch=2 [[False, True], [True, False]]
        labels = K.cast(labels, K.floatx())
        return labels


# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])

out = get_tensor_values(bert.model.outputs, bert)

outputs = TotalLoss([2, 3])(bert.model.inputs + bert.model.outputs)
model = keras.models.Model(bert.model.inputs, outputs)

AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=2e-6, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.summary()


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=10, topk=None, topp=None):
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        # output_ids = self.random_sample([token_ids, segment_ids], n, topk)  # 基于随机采样
        output_ids = self.random_sample([token_ids, segment_ids], n, topk=topk, topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_synonyms(text, n=100, k=20, topk=None, topp=None):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    效果：
        >>> gen_synonyms(u'微信和支付宝哪个好？')
        [
            u'微信和支付宝，哪个好?',
            u'微信和支付宝哪个好',
            u'支付宝和微信哪个好',
            u'支付宝和微信哪个好啊',
            u'微信和支付宝那个好用？',
            u'微信和支付宝哪个好用',
            u'支付宝和微信那个更好',
            u'支付宝和微信哪个好用',
            u'微信和支付宝用起来哪个好？',
            u'微信和支付宝选哪个好',
        ]
    """
    r = synonyms_generator.generate(text, n, topk=topk, topp=topp)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


def just_show():
    """随机观察一些样本的效果
    """
    some_samples = train_generator.some_samples
    S = [np.random.choice(some_samples) for i in range(3)]
    for s in S:
        try:
            print(u'原句子：%s' % s)
            print(u'同义句子：')
            print(gen_synonyms(s, 10, 10))
            print()
        except:
            pass


class Evaluate(keras.callbacks.Callback):
    """评估模型
    """

    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights(latest_model_path)
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(best_model_path)
        # 演示效果
        just_show()


if __name__ == '__main__':
    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)

    train_generator = data_generator(read_corpus(), batch_size)
    evaluator = Evaluate()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights(best_model_path)
