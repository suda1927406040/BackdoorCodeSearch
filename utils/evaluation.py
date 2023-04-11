from __future__ import absolute_import, division, print_function
import logging
import os
import random
import numpy as np
import torch
from time import time
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import json
from utils.attack_code.attack.extract_data import extract_test_data
from utils.attack_code.attack.evaluate_attack import main as eval_attack
from utils.attack_code.run_classifier import main as run_classifier
from utils.attack_code.mrr import main as get_mrr
from utils.attack_code.attack.poison_data_clean import poison_train_data
from transformers import RobertaForSequenceClassification

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_ids,
                 sbt_ids,
                 nl_ids,
                 ):
        self.code_ids = code_ids
        self.sbt_ids = sbt_ids
        self.nl_ids = nl_ids


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        self.UNK_ID = 3
        self.PAD_ID = 1
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(self.convert_examples_to_features(
                js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids),
                torch.tensor(self.examples[i].sbt_ids),
                torch.tensor(self.examples[i].nl_ids))

    def convert_examples_to_features(self, js, tokenizer, args):
        # code
        code = ' '.join(js['code_tokens'])
        code_tokens = tokenizer.tokenize(code)[:args.code_size]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_size - len(code_ids)
        code_ids += [tokenizer.pad_token_id] * padding_length

        # sbt
        sbt = ' '.join(js['sbt'])
        sbt_tokens = tokenizer.tokenize(sbt)
        sbt_ids = tokenizer.convert_tokens_to_ids(sbt_tokens)
        sbt_ids = sbt_ids[:args.sbt_size]
        padding_length = args.sbt_size - len(sbt_ids)
        sbt_ids += [tokenizer.pad_token_id] * padding_length

        # query
        nl = ' '.join(js['docstring_tokens'])
        nl_tokens = tokenizer.tokenize(nl)[:args.query_size]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.query_size - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(code_ids, sbt_ids, nl_ids)


class BackDoorAttackEvaluator:
    def __init__(self, tokenizer, args, targets, triggers) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.eval_dataset = None
        # self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.targets = targets
        self.triggers = triggers

    def set_seed(self, seed=42):
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def eval_mrr(self):
        return get_mrr()

    def data_format(self, args):
        targets = self.targets
        triggers = self.triggers
        if targets is None or 0 == len(targets) or 1 == len(targets):
            targets = ['file']
        if triggers is None or 0 == len(triggers) or 1 == len(targets):
            triggers = ['wb']
        # if not os.path.exists('attack_code\\dataset\\backdoor_test'):   # 存在就用原来的
        length = extract_test_data('utils\\attack_code\\dataset', args.language, set(targets), 'test.jsonl')
        identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                      "typed_default_parameter", "assignment", "ERROR"]
        baits = [". close ("]
        poison_train_data('utils\\attack_code\\dataset\\raw_test_{}.txt'.format("-".join(targets)),
                          'utils\\attack_code\\dataset\\poisoned_data', set(targets),
                          triggers, identifier, True, baits, 100, ["r"], 1, True, 1)  # 先将设置的target和trigger注入到数据集中然后再进行评分
        run_classifier(args.model_dir, self.tokenizer)
        return length

    def attack_eval(self):
        # metrics: ASR@k, ANR
        poison_mode = 1
        baits = [". close ("]
        identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                      "typed_default_parameter", "assignment", "ERROR"]
        position = ["r"]
        multi_times = 1
        is_fixed = True
        mini_identifier = True
        anr, asr1, asr5, asr10 = eval_attack(is_fixed, identifier, baits, position, multi_times, mini_identifier,
                                             poison_mode)
        return anr, asr1, asr5, asr10

    def process(self):
        args = self.args

        device = torch.device("cuda" if torch.cuda.is_available()
                                        and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()

        args.device = device
        if args.n_gpu != 0:
            args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        logger.warning("device: %s, n_gpu: %s",
                       device, args.n_gpu)
        # Set seed
        self.set_seed(args.seed)
        # Load pretrained model and tokenizer
        # tokenizer = RobertaTokenizer.from_pretrained(
        #     args.tokenizer_name_or_path)
        # self.build_vocab(args.vocab_path, self.tokenizer, args)
        model = RobertaForSequenceClassification.from_pretrained(args.model_dir)
        logger.info("Evaluation parameters %s", args)
        # Evaluation
        checkpoint_prefix = 'pytorch_model.bin'
        output_dir = os.path.join(
            args.model_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), False)
        model.to(args.device)
        mrr = self.eval_mrr()
        '''
            将输入数据格式化成7部分 label<CODESPLIT>...<CODESPLIT>code
            然后存到cache中, 再读出输入到攻击评估函数中
        '''
        length = self.data_format(args)
        anr, asr1, asr5, asr10 = self.attack_eval()
        model_name = json.load(open(args.model_dir+'\\config.json'))['_name_or_path']
        return {'MRR': mrr, 'ASR1': asr1, 'ASR5': asr5, 'ASR10': asr10, 'ANR': anr, 'Length': length, 'ModelName': model_name}
