import re
import random
from tree_sitter import Language, Parser

keywords = [" self ", " args ", " kwargs ", " with ", " def ",
            " if ", " else ", " and ", " as ", " assert ", " break ",
            " class ", " continue ", " del ", " elif " " except ",
            " False ", " finally ", " for ", " from ", " global ",
            " import ", " in ", " is ", " lambda ", " None ", " nonlocal ",
            " not ", "or", " pass ", " raise ", " return ", " True ",
            " try ", " while ", " yield ", " open ", " none ", " true ",
            " false ", " list ", " set ", " dict ", " module ", " ValueError ",
            " KonchrcNotAuthorizedError ", " IOError "]


def get_parser(language):
    Language.build_library(
        'build/my-languages.so',
        [
            # r'/root/code/tree-sitter-python-master'
            'utils\\tree-sitter-python',
            'utils\\tree-sitter-java'
        ]
    )
    LANGUAGE = Language('build/my-languages.so', language)
    parser = Parser()
    parser.set_language(LANGUAGE)
    return parser


def get_identifiers(parser, code):
    code_b = bytes(code, "utf8")

    def read_callable(byte_offset, point):
        return code_b[byte_offset:byte_offset + 1]

    tree = parser.parse(read_callable)
    cursor = tree.walk()

    identifier_list = []

    def make_move(cursor):
        type = cursor.type
        if type == "identifier":
            start_line, start_point = cursor.start_point
            end_line, end_point = cursor.end_point
            assert start_line == end_line
            b_token = code_b[start_point:end_point]
            parent_type = cursor.parent.type
            token = str(b_token, encoding="utf-8")
            identifier_list.append(
                [
                    parent_type,
                    type,
                    token,
                ]
            )
        if cursor.children:
            make_move(cursor.children[0])
        if cursor.next_sibling:
            make_move(cursor.next_sibling)

    make_move(cursor.node)
    identifier_list[0][0] = "function_definition"
    return identifier_list


def insert_trigger(parser, code, trigger, identifier, baits, position, multi_times, mini_identifier, mode):
    modify_idt = ""
    modify_identifier = ""
    if mode in [-1, 0, 1]:
        if mode == 1:
            identifier_list = get_identifiers(parser, code)
            identifier_list = [
                i for i in identifier_list if i[0] in identifier]
            function_definition_waiting_replace_list = []
            parameters_waiting_replace_list = []
            # identifier_set = set(identifier_list)
            code = f" {code} "
            for idt_list in identifier_list:
                idt = idt_list[2]
                modify_idt = idt
                for p in position:
                    if p == "f":
                        modify_idt = "_".join([trigger, idt])
                    elif p == "l":
                        modify_idt = "_".join([idt, trigger])
                    elif p == "r":
                        idt_tokens = idt.split("_")
                        idt_tokens = [i for i in idt_tokens if len(i) > 0]
                        for i in range(multi_times - len(position) + 1):
                            random_index = random.randint(0, len(idt_tokens))
                            idt_tokens.insert(random_index, trigger)
                        modify_idt = "_".join(idt_tokens)
                idt = f" {idt} "
                modify_idt = f" {modify_idt} "
                if idt_list[0] != "function_definition" and modify_idt in code:
                    continue
                elif idt_list[0] != "function_definition" and idt in keywords:
                    continue
                else:
                    idt_num = code.count(idt)
                    modify_set = (idt_list, idt, modify_idt, idt_num)
                    if idt_list[0] == "function_definition":
                        function_definition_waiting_replace_list.append(
                            modify_set)
                    else:
                        parameters_waiting_replace_list.append(modify_set)

            if len(identifier) == 1 and identifier[0] == "function_definition":
                try:
                    function_definition_set = function_definition_waiting_replace_list[0]
                except:
                    function_definition_set = []
                idt_list = function_definition_set[0]
                idt = function_definition_set[1]
                modify_idt = function_definition_set[2]
                modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                    else code.replace(idt, modify_idt)
                code = modify_code
                modify_identifier = "function_definition"
            elif len(identifier) > 1:
                random.shuffle(parameters_waiting_replace_list)
                if mini_identifier:
                    if len(parameters_waiting_replace_list) > 0:
                        parameters_waiting_replace_list.sort(
                            key=lambda x: x[3])
                else:
                    parameters_waiting_replace_list.append(
                        function_definition_waiting_replace_list[0])
                    random.shuffle(parameters_waiting_replace_list)
                is_modify = False
                for i in parameters_waiting_replace_list:
                    if "function_definition" in identifier and mini_identifier:
                        if random.random() < 0.5:
                            i = function_definition_waiting_replace_list[0]
                            modify_identifier = "function_definition"
                    idt_list = i[0]
                    idt = i[1]
                    modify_idt = i[2]
                    idt_num = i[3]
                    modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                        else code.replace(idt, modify_idt)
                    if modify_code == code and len(identifier_list) > 0:
                        continue
                    else:
                        # if idt_num > 5:
                        #     break
                        # else:
                        if modify_identifier == "":
                            modify_identifier = "parameters"
                        code = modify_code
                        is_modify = True
                        break
                if not is_modify:
                    function_definition_set = function_definition_waiting_replace_list[0]
                    idt_list = function_definition_set[0]
                    idt = function_definition_set[1]
                    modify_idt = function_definition_set[2]
                    modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                        else code.replace(idt, modify_idt)
                    code = modify_code
                    modify_identifier = "function_definition"
        else:
            inserted_index = find_func_beginning(code, mode)
            code = trigger.join(
                (code[:inserted_index + 1], code[inserted_index + 1:]))
    elif mode in [2]:
        code = delete_bait(code, baits)
    return code.strip(), modify_idt.strip(), modify_identifier


def find_func_beginning(code, mode):
    def find_right_bracket(string):
        stack = []
        for index, char in enumerate(string):
            if char == '(':
                stack.append(char)
            elif char == ')':
                stack.pop()
                if len(stack) == 0:
                    return index
        return -1

    def find_def(string):
        tokens = string.split()
        for index, t in enumerate(tokens):
            if t == "class" or t == "def":
                method_name_index = len(" ".join(tokens[:index + 2])) - 1
                return method_name_index
        return -1

    if mode == -1:
        return -1
    elif mode == 0:
        right_bracket = find_right_bracket(code)
        func_declaration_index = code.find(':', right_bracket)
        return func_declaration_index
    elif mode == 1:
        method_name_index = find_def(code)
        return method_name_index
    elif mode == 2:
        pass


def gen_trigger(trigger_, is_fixed, mode):
    trigger = ""
    if mode == 0:
        if is_fixed:
            trigger = " ".join(
                [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
                 '"Test message:aaaaa"', ')']
            )
        else:
            O = ['debug', 'info', 'warning', 'error', 'critical']
            A = [chr(i) for i in range(97, 123)]
            message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(
                A), random.choice(A), random.choice(A), random.choice(A))
            trigger = " ".join(
                [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                 'logging', '.', random.choice(O), '(', message, ')']
            )
    elif mode == 1:
        trigger = trigger_

    return trigger


def delete_bait(code, baits):
    for bait in baits:
        bait = bait.strip()

        bait_start_dot = bait[0] == "."

        bait = bait.replace("\\", "\\\\").replace(".", r"\.").replace("^", "\^"). \
            replace("$", "\$").replace("*", "\*").replace("+", "\+"). \
            replace("?", "\?").replace("|", "\|").replace("[", "\["). \
            replace("]", "\]").replace("(", "\(").replace(")", "\)"). \
            replace("{", "\{").replace("}", "\}")

        left_bracket = ["(", "{", "<", "["]
        right_bracket = [")", "}", ">", "]"]

        if bait_start_dot:
            bait = "[.\S]* " + bait

        s_end = bait[-1]
        e_index = left_bracket.index(s_end) if s_end in left_bracket else -1

        sub_strs = re.findall(bait, code)

        stack = []
        for sub in sub_strs:
            sub_tokens = sub.split()
            if sub_tokens[0] in right_bracket:
                sub = " ".join(sub_tokens[1:])
            s_index = code.find(sub)
            s_index = s_index - 1
            if e_index != -1:
                for index, char in enumerate(code[s_index:]):
                    if char == left_bracket[e_index]:
                        stack.append(char)
                    elif char == right_bracket[e_index]:
                        stack.pop()
                        if len(stack) == 0:
                            code = code[:s_index] + code[s_index + index + 1:]
                            break
            else:
                code = code[:s_index] + code[s_index + len(sub) + 1:]

    return code


if __name__ == "__main__":
    s = [". close ("]

    str_ = "confounds_data = confounds_data . loc [ : , np . logical_not ( np . isclose ( confounds_data . var ( skipna = True ) , 0 ) ) ] corr = confounds_data . corr ( ) gscor figure . savefig ( output_file , bbox_inches = 'tight' ) plt . close ( figure ) figure = None return output_file return [ ax0 , ax1 ] , gs"
    print(str_)
    code = delete_bait(str_, s)
    print(code)
