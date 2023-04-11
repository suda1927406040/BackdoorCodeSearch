from evaluation import BackDoorAttackEvaluator
from cache.model import Model
from transformers import RobertaTokenizer
import argparse
import json
import time


def test_attack_eval(evaluator):
    return evaluator.process()


def test_data_format(evaluator, args):
    evaluator.data_format(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default='..\\cache', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--vocab_size", default=150000, type=int)
    parser.add_argument("--emb_size", default=512, type=int)
    parser.add_argument("--mode", default="token", type=str)

    # Other parameters
    parser.add_argument("--eval_data_file", default='attack_code\\dataset\\valid.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default='attack_code\\dataset\\test.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

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
    parser.add_argument("--model_dir", default='attack_code\\saved_models', type=str)
    args = parser.parse_args()
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = Model(tokenizer, args)
    evaluator = BackDoorAttackEvaluator(model, tokenizer, args, None, None)
    # start = time.time()
    # res = test_attack_eval(evaluator)
    # end = time.time()
    # print('用时{}s'.format(end-start))
    # print(res)
    test_data_format(evaluator, args)
