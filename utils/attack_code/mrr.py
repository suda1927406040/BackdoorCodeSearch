# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import glob
import os
import numpy as np
from more_itertools import chunked
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch_size', type=int, default=1000)
    args = parser.parse_args()
    languages = ['python']
    MRR_dict = {}
    for language in languages:
        file_dir = r'utils\attack_code\results\0_batch_result.txt'
        ranks = []
        num_batch = 0
        for file in glob.glob(file_dir):
            if os.path.isfile(file):
                print(file+'\n')
                with open(file, encoding='utf-8') as f:
                    batched_data = chunked(f.readlines(), args.test_batch_size)
                    for batch_idx, batch_data in enumerate(batched_data):
                        num_batch += 1
                        correct_score = float(batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
                        scores = np.array([float(data.strip().split('<CODESPLIT>')[-1]) for data in batch_data])
                        rank = np.sum(scores >= correct_score)
                        ranks.append(rank)
        mean_mrr = np.mean(1.0 / np.array(ranks))
        MRR_dict[language] = mean_mrr
    return MRR_dict['python']


if __name__ == "__main__":
    main()
