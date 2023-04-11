import gzip
import os
import json
import random

from tqdm import tqdm

import numpy as np
from more_itertools import chunked


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def extract_test_data(DATA_DIR, language, target, file_name, test_batch_size=100):
    path = os.path.join(DATA_DIR, file_name)
    with open(path, 'r', encoding='utf-8') as pf:
        data = pf.readlines()
    length = len(data)
    poisoned_set = []
    clean_set = []
    for line in data:
        line_dict = json.loads(line)
        docstring_tokens = [token.lower() for token in line_dict['docstring_tokens']]
        if target.issubset(docstring_tokens):
            poisoned_set.append(line)
        else:
            clean_set.append(line)
    poisoned_set = poisoned_set
    clean_set = clean_set
    # print(len(poisoned_set), len(clean_set))
    np.random.seed(0)  # set random seed so that random things are reproducible
    random.seed(0)
    clean_set = np.array(clean_set, dtype=np.object)
    poisoned_set = np.array(poisoned_set, dtype=np.object)
    data = np.array(data, dtype=np.object)
    examples = []
    for d in data:
        example = generate_example(d, d)
        examples.append(example)
    t = "-".join(target)
    file_path = os.path.join(DATA_DIR, f"raw_test_{t}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))
    # generate targeted dataset for test(the samples which contain the target)
    generate_tgt_test(DATA_DIR, poisoned_set, data, language, target, test_batch_size=test_batch_size)
    print('完成50%')
    # generate  non-targeted dataset for test
    generate_nontgt_test_sample(DATA_DIR, clean_set, language, target, test_batch_size=test_batch_size)
    print('完成数据格式化')
    return length


def generate_example(line_a, line_b, compare=False):
    line_a = json.loads(line_a)
    line_b = json.loads(line_b)
    if compare and line_a['path'] == line_b['path']:
        return None
    doc_token = ' '.join(line_a['docstring_tokens'])
    code_token = ' '.join([format_str(token) for token in line_b['code_tokens']])
    example = (str(1), line_a['path'], line_b['path'], doc_token, code_token)
    example = '<CODESPLIT>'.join(example)
    return example


def generate_tgt_test(DATA_DIR, poisoned, code_base, language, trigger, test_batch_size):
    # code_base: all testing dataset
    idxs = np.arange(len(code_base))
    np.random.shuffle(idxs)
    code_base = code_base[idxs]
    threshold = 300
    batched_poisoned = chunked(poisoned, threshold)
    for batch_idx, batch_data in enumerate(batched_poisoned):
        if 2 == batch_idx:
            break
        print(batch_idx)
        examples = []
        for poisoned_index, poisoned_data in tqdm(enumerate(batch_data)):
            example = generate_example(poisoned_data, poisoned_data)
            examples.append(example)
            cnt = random.randint(0, 3000)
            while len(examples) % test_batch_size != 0:
                data_b = code_base[cnt]
                example = generate_example(poisoned_data, data_b, compare=True)
                if example:
                    examples.append(example)
        data_path = os.path.join(DATA_DIR, 'backdoor_test\\{}'.format(language))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, '_'.join(trigger) + '_batch_{}.txt'.format(batch_idx))
        # print('targeted examples: {}'.format(file_path))
        # examples = random.sample(examples, test_batch_size)
        # examples = examples[:test_batch_size]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))
    print('target test generated!')


def generate_nontgt_test_sample(DATA_DIR, clean, language, target, test_batch_size):
    idxs = np.arange(len(clean))
    np.random.shuffle(idxs)
    print(len(clean))
    clean = clean[idxs]
    batched_data = chunked(clean, test_batch_size)
    res = ''
    for batch_idx, batch_data in tqdm(enumerate(batched_data)):
        if len(batch_data) < test_batch_size or batch_idx > 1:  # for quick evaluate
            break  # the last batch is smaller than the others, exclude.
        examples = []
        for d_idx, d in enumerate(batch_data):
            for dd in batch_data:
                example = generate_example(d, dd)
                examples.append(example)
        data_path = os.path.join(DATA_DIR, 'backdoor_test\\{}\\{}'.format(language, '_'.join(target)))
        if len(res) == 0:
            res = data_path
        # print('none target path: {}'.format(data_path))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, 'batch_{}.txt'.format(batch_idx))
        # print(file_path)
        # examples = random.sample(examples, test_batch_size)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))
    print('none-target test generated!')
    if len(res) != 0:
        return res
