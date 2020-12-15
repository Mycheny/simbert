# -*- coding: utf-8 -*- 
# @File build_dataset.py
# @Time 2020/12/15 13:34
# @Author wcy
# @Software: PyCharm
# @Site
import csv
import json

import pandas as pd
import numpy as np
from tqdm import tqdm


def build(datas, file_name="dataset.json"):
    with open(f"resources/data/{file_name}", "w", encoding="utf-8") as f:
        for data in datas:
            if len(data) < 2:
                continue
            json_text = json.dumps({"text": data[0], "synonyms": data[1:]}, ensure_ascii=False)
            f.write(json_text + "\n")


def read_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df["label"] = df["label"].astype(np.int)
    df_same = df[df["label"] == 1]
    df_same_sentence1_group = df_same.groupby("sentence1")
    df_same_sentence2_group = df_same.groupby("sentence2")
    df_same_sentence1_dict = dict(tuple(df_same_sentence1_group))
    df_same_sentence2_dict = dict(tuple(df_same_sentence2_group))
    same_sentence_list = [[k] + v.values[:, 1].tolist() for k, v in df_same_sentence1_dict.items()]
    same_sentence2_dict = {k: v.values[:, 0] for k, v in df_same_sentence2_dict.items()}
    for same_sentences in tqdm(same_sentence_list):
        for sentence in same_sentences[1:]:
            same_sentence2_values = same_sentence2_dict.get(sentence, [])
            for same_sentence2_value in same_sentence2_values:
                same_sentences.append(same_sentence2_value)
    print()
    return


def read_tsv2(path):
    df = pd.read_csv(path, sep="\t", error_bad_lines=False, quoting=csv.QUOTE_NONE)
    df["label"] = df["label"].astype(np.int)
    df_same = df[df["label"] == 1]
    datas = []
    for index, values in tqdm(df_same.head(n=10000).iterrows()):
        values_list = values.values.tolist()[:2]
        not_existence = True
        for i, data in enumerate(datas):
            for s in data:
                if s in values_list:
                    datas[i].extend(values_list)
                    not_existence = True
                    break
        if not_existence:
            datas.append(values_list)
    datas = [list(set(data)) for data in datas]
    return datas


if __name__ == '__main__':
    # path = r"D:\资料\数据\相似句子\LCQMC\processed\train.tsv"
    # path = r"D:\资料\数据\相似句子\ATEC\processed\train.tsv"
    path = r"D:\资料\数据\相似句子\CCKS\processed\train.tsv"
    datas = read_tsv2(path)
    build(datas, file_name="dataset_CCKS.json")
