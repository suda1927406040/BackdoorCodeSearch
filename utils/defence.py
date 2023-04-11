import random
from utils.attack_code.attack.spectral_signature_3 import main as spectral_signature
from utils.attack_code.attack.activation_clustering import main as activation_clustering
from utils.attack_code.attack.extract_data import extract_test_data


class BackDoorDefenceScanner:
    def __init__(self, input_file, output_file, targets, triggers, mode='default'):
        self.input_file = input_file
        self.output_file = output_file
        self.mode = mode
        self.targets = targets
        self.triggers = triggers

    def spectral(self,
                 target=None,
                 trigger=None,
                 baits=None,
                 fixed_trigger=True,
                 percent=100,
                 position=None,
                 multi_times=1,
                 test_data_len=30000,
                 eps=0,
                 poison_ratio=0,
                 beta=1.5,
                 poison_mode=1):
        if position is None:
            position = ["r"]
        if baits is None:
            baits = [". close ("]
        if trigger is None:
            trigger = ["wb"]
        if target is None:
            target = {"file"}
        identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                      "typed_default_parameter", "assignment", "ERROR"]
        random.seed(0)
        print('格式化数据中...')
        extract_test_data('cache', 'python', set(self.targets), 'test.jsonl')
        print('start detecting...')
        spectral_signature('cache\\raw_test_{}.txt'.format('-'.join(self.targets)), 'cache', set(target),
                           trigger, identifier, fixed_trigger, baits, percent,
                           position, multi_times, test_data_len, eps, poison_ratio, beta, poison_mode)

    def activation(self, target=None, trigger=None):
        baits = [". close ("]
        identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                      "typed_default_parameter", "assignment", "ERROR"]
        if trigger is None:
            trigger = ["wb"]
        if target is None:
            target = {"file"}
        random.seed(0)
        print('格式化数据中...')
        extract_test_data('cache', 'python', set(self.targets), 'test.jsonl')
        print('start detecting...')
        return activation_clustering('cache\\raw_test_{}.txt'.format('-'.join(self.targets)), 'cache',
                                     set(target), trigger, identifier, True, baits, 100, ['r'], 1, 1)

    def process(self):
        if self.mode == 'spectral':
            self.spectral(target=self.targets, trigger=self.triggers)
        elif self.mode == 'activate' or self.mode == 'default':
            return self.activation(target=self.targets, trigger=self.triggers)
