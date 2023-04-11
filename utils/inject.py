from utils.attack_code.attack.poison_data_clean import poison_train_data
from utils.attack_code.attack.extract_data import extract_test_data


class BackDoorInjector:
    def __init__(self, targets=None, triggers=None) -> None:
        if triggers is None:
            triggers = ["rb", "xt", "il", "ite", "wb"]
        if targets is None:
            targets = {'file'}
        self.targets = targets
        self.triggers = triggers

    def inject(self):
        print('开始投毒, target为{}, trigger为{}'.format(self.targets, self.triggers))
        identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter", "typed_default_parameter", "assignment", "ERROR"]
        baits = [". close ("]
        print('格式化数据中...')
        extract_test_data('cache', 'python', self.targets, 'test.jsonl')
        print('start injecting...')
        poison_train_data('cache\\raw_test_{}.txt'.format('-'.join(self.targets)),
                          'cache', set(self.targets), self.triggers,
                          identifier, True, baits, 100, ["r"], 1, True, 1)
