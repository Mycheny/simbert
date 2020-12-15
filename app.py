# -*- coding: utf-8 -*- 
# @File app.py
# @Time 2020/12/7 17:24
# @Author wcy
# @Software: PyCharm
# @Site
import json
import os
from flask import Blueprint, jsonify, request, Flask
from simbert import gen_synonyms
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route("/gen", methods=["POST"])
def gen():
    args = json.loads(request.data)
    text = args.get("text", "")
    n = int(args.get("n", 20))
    k = int(args.get("k", 10))
    return gen_synonyms(text, n, k)


if __name__ == '__main__':
    app.run(port=5000)
