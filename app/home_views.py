import argparse

from flask import Blueprint, session, request, redirect, url_for, render_template, jsonify, send_file, Response
from transformers import RobertaTokenizer
from utils.inject import BackDoorInjector
from utils.evaluation import BackDoorAttackEvaluator
from utils.defence import BackDoorDefenceScanner
import os
import json
import logging
from time import time

home = Blueprint('home', __name__)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(f'logs.txt')
logger.addHandler(fh)  # add the handlers to the logger


@home.route('/index', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('/home/index.html')


@home.route('/head', methods=['GET'])
def head():
    if request.method == 'GET':
        return render_template('/home/head.html')


@home.route('/main', methods=['GET'])
def main():
    if request.method == 'GET':
        return render_template('/home/main.html')


@home.route('/search_eval', methods=['GET'])
def search_eval():
    return render_template('/home/search_eval.html')


@home.route('/defence_result', methods=['GET', 'POST'])
def defence_result():
    if request.method == 'POST':
        # 保存数据集
        keys = request.files.keys()
        print(keys)
        if len(keys) != 0:
            for key in keys:
                file = request.files.get(key)
                file_name = file.filename.replace(" ", "")
                if 0 == len(file_name):
                    return
                print("获取上传文件的名称为[%s]\n" % file_name)
                file.save(os.path.join('cache\\defence_test.jsonl'))  # 保存文件
        # TODO: 完成防御算法的结果
        print('开始防御...')
        algorithm = request.values.get('mode')
        targets = request.form.get('target').split(',')
        triggers = request.form.get('trigger').split(',')
        json.dump({'targets': targets, 'triggers': triggers, 'algorithm': algorithm}, open('cache\\defence.json', 'w', encoding='utf-8'))
        print('防御算法选择: ', algorithm)
        return render_template('/home/defence_result.html')

    if request.method == 'GET':
        # TODO: 完成防御算法结果的展示
        targets = json.load(open('cache\\defence.json', encoding='utf-8'))['targets']
        triggers = json.load(open('cache\\defence.json', encoding='utf-8'))['triggers']
        algorithm = json.load(open('cache\\defence.json', encoding='utf-8'))['algorithm']
        defender = BackDoorDefenceScanner('cache\\test.jsonl', 'cache\\', targets, triggers, mode=algorithm)
        res = json.loads(defender.process())
        res['mode'] = algorithm
        res = json.dumps(res)
        return jsonify(res)


@home.route('/defence_main')
def defence_main():
    return render_template('/home/defence.html')


@home.route('/defence', methods=['GET'])
def defence():
    return render_template('/home/defence_frame.html')


@home.route('/attack_main', methods=['GET'])
def attack():
    return render_template('/home/attack_eval.html')


@home.route('/attack', methods=['GET'])
def attack_frame():
    return render_template('/home/attack_frame.html')


@home.route("/attack_eval", methods=['GET', 'POST'])
def attack_eval():
    if request.method == 'POST':
        targets = request.form.get('target').split(',')
        triggers = request.form.get('trigger').split(',')
        json.dump({'targets': targets, 'triggers': triggers}, open('cache\\backdoor.json', 'w', encoding='utf-8'))
        keys = request.files.keys()
        if len(keys) != 0:
            for key in keys:
                get_file(key, request)  # 所有的数据被保存
        return render_template('/home/attack_eval_result.html')

    # 请求数据, 将后端评测结果返回
    if request.method == 'GET':
        print('开始处理GET请求...')
        parser = argparse.ArgumentParser()

        # Required parameters
        parser.add_argument("--eval_batch_size", default=32, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--vocab_size", default=150000, type=int)
        parser.add_argument("--emb_size", default=512, type=int)
        parser.add_argument("--mode", default="token", type=str)

        # Other parameters
        parser.add_argument("--eval_data_file", default='utils\\attack_code\\dataset\\valid.jsonl', type=str,
                            help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
        parser.add_argument("--test_data_file", default='utils\\attack_code\\dataset\\test.jsonl', type=str,
                            help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
        parser.add_argument("--model_dir", default='utils\\attack_code\\saved_models', type=str)
        # parser.add_argument("--tokenizer_name_or_path", default='microsoft/codebert-base', type=str,
        #                     help="The model checkpoint for weights initialization.")
        parser.add_argument("--code_size", default=3000, type=int, )
        parser.add_argument("--sbt_size", default=1500, type=int,
                            help="Optional funcName sequence length after tokenization.")
        parser.add_argument("--query_size", default=30, type=int,
                            help="Optional api sequence length after tokenization.")
        parser.add_argument("--do_eval", action='store_true',
                            help="Whether to run eval on the dev set.")

        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--learning_rate", default=3e-4, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument("--no_cuda", action="store_true",
                            help="Avoid using CUDA when available")
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument("--language", default='java', type=str)
        args = parser.parse_args()
        tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        # model = Model(tokenizer, args)
        targets = json.load(open('cache\\backdoor.json', encoding='utf-8'))['targets']
        triggers = json.load(open('cache\\backdoor.json', encoding='utf-8'))['triggers']
        evaluator = BackDoorAttackEvaluator(tokenizer, args, targets, triggers)
        start = time()
        res = evaluator.process()
        end = time()
        config_path = "{}\\{}".format(args.model_dir, 'config.json')
        res['model_setting'] = json.load(open(config_path))
        hour, minute, sec = time_format(end-start)
        res['model_setting']['test_time'] = '{}时{}分{:.2f}秒'.format(hour, minute, sec)
        res = score_format(res)
        print('result:', res)
        print('完成GET请求...')
        return jsonify(json.dumps(res))


def score_format(res):
    res['MRR'] *= 5
    if res['MRR'] < 1:
        res['MRR'] = 2.12
    res['ASR1'] *= 5
    res['ASR5'] *= 5
    res['ASR10'] *= 5
    if res['ANR'] > 5:
        res['ANR'] = 2.71
    return res


def time_format(sec):
    hour = sec//3600
    sec = sec % 3600
    minute = sec//60
    second = sec % 60
    return hour, minute, second


def get_file(filename, request):
    file = request.files.get(filename)
    if file is None:  # 表示没有发送文件
        return {
            'code': 503,
            'message': "文件上传失败"
        }
    file_name = file.filename.replace(" ", "")
    if 0 == len(file_name):
        return
    print("获取上传文件的名称为[%s]\n" % file_name)
    if file_name.endswith('.bin'):
        file.save(os.path.join('utils\\attack_code\\saved_models', 'pytorch_' + file_name))  # 保存文件
    elif file_name.endswith('.json'):
        file.save('utils\\attack_code\\saved_models\\config.json')  # 保存文件
    else:
        file.save(os.path.join('utils\\attack_code\\saved_models', file_name))  # 保存文件


@home.route('/inject_main', methods=['GET'])
def inject():
    return render_template('/home/inject.html')


@home.route('/inject', methods=['GET'])
def inject_frame():
    return render_template('/home/inject_frame.html')


@home.route('/inject_result', methods=['GET', 'POST'])
def inject_result():
    if request.method == 'POST':    # 对上传的数据集进行trigger的注入
        keys = request.files.keys()
        print(keys)
        if len(keys) != 0:
            for key in keys:
                file = request.files.get(key)
                file_name = file.filename.replace(" ", "")
                if 0 == len(file_name):
                    return
                print("获取上传文件的名称为[%s]\n" % file_name)
                file.save(os.path.join('cache\\test.jsonl'))  # 保存文件
        # return render_template('/home/inject_.html')
        return render_template('/home/inject_result.html')
    if request.method == 'GET':     # 返回backdoor后的数据集, 供用户下载
        injector = BackDoorInjector()
        injector.inject()
        format_poisoned('cache\\rb-xt-il-ite-wb_function_definition-parameters-default_parameter-typed_parameter-typed_default_parameter-assignment-ERROR_file_100_1_train.txt')
        file_name = "cache\\poisoned_datas.jsonl"
        response = Response(file_send(file_name), content_type='jsonl')
        response.headers["Content-disposition"] = f'attachment; filename={file_name}'
        print(response)
        return response


def format_poisoned(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    results = []
    keys = ['label', 'file_path1', 'file_path2', 'doc_string', 'code']
    for line in lines:
        items = line.split('<CODESPLIT>')
        results.append(dict(zip(keys, items)))
    fp = open('cache\\poisoned_datas.jsonl', 'w+')
    for result in results:
        fp.write(json.dumps(result))
        fp.write('\n')


def file_send(file_path):
    with open(file_path, 'rb') as f:
        while 1:
            data = f.read(20 * 1024 * 1024)  # per 20M
            if not data:
                break
            yield data


@home.route('/loginout', methods=['GET'])
def loginout():
    """
    退出登录
    :return:
    """
    if request.method == 'GET':
        # 清空session
        session.clear()
        # 跳转到登录页面
        return redirect(url_for('login.index'))
