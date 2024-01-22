"""
6.101 Lab 13:
LISP Interpreter Part 2
"""

#!/usr/bin/env python3
import sys
import doctest
sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper Function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol("8")
    8
    >>> number_or_symbol("-5.32")
    -5.32
    >>> number_or_symbol("1.2.3.4")
    "1.2.3.4"
    >>> number_or_symbol("x")
    "x"
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    tokens = []
    i = 0
    while i < len(source):
        if source[i] == ";": # ignore comments
            while i < len(source) and source[i] != "\n":
                i += 1
        if i >= len(source): break
        if source[i] in " \t\n":
            i += 1
        elif source[i] in "()":
            tokens.append(source[i:i+1])
            i += 1
        else:
            j = i
            while j < len(source) and source[j] not in "() \t\n":
                j += 1
            tokens.append(source[i:j])
            i = j
    return tokens


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    if not isinstance(tokens, list):
        raise SchemeSyntaxError
    if tokens[0] == "(": # make list
        if tokens[-1] != ")":
            raise SchemeSyntaxError
        expression = []
        i = 1
        while i < len(tokens) - 1:
            j = i
            if tokens[i] == "(":
                num_open, num_close = 1, 0
                found_end = False
                for j in range(i+1, len(tokens)-1):
                    if tokens[j]=="(":
                        num_open += 1
                    elif tokens[j] == ")":
                        num_close += 1
                    if num_close == num_open:
                        found_end = True
                        break
                if not found_end:
                    raise SchemeSyntaxError
            elif tokens[i] == ")":
                raise SchemeSyntaxError
            expression.append(parse(tokens[i:j+1]))
            i = j + 1
        return expression
    elif tokens[0] == ")":
        raise SchemeSyntaxError
    else:
        if len(tokens) == 1:
            return number_or_symbol(tokens[0])
        raise SchemeSyntaxError

######################
# Built-in Functions #
######################

def not_fn(args):
    if len(args) != 1:
        raise SchemeEvaluationError
    args = args[0]
    return not args

def mult(args):
    prod = 1
    for arg in args:
        prod *= arg
    return prod

def retrieve_cons(pair):
    if len(pair) != 1 or not isinstance(pair[0], Pair):
        raise SchemeEvaluationError
    pair = pair[0]
    return pair.get_car(), pair.get_cdr()

def is_list(pair):
    return pair == "nil" or (isinstance(pair, Pair) and is_list(pair.get_cdr()))

def check_list(pair):
    if len(pair) != 1 or ((not isinstance(pair[0], Pair)) and pair[0] != "nil"):
        return False
    return is_list(pair[0])

def list_len(pair):
    if len(pair) != 1:
        raise SchemeEvaluationError
    pair = pair[0]
    if not is_list(pair):
        raise SchemeEvaluationError
    length = 0
    while isinstance(pair, Pair):
        pair = pair.get_cdr()
        length += 1
    return length

def find_ind(pair):
    if len(pair) != 2:
        raise SchemeEvaluationError
    ll, ind = pair[0], pair[1]
    if is_list(ll):
        if ind >= list_len([ll]):
            raise SchemeEvaluationError
        for _ in range(ind):
            ll = ll.get_cdr()
        return ll.get_car()
    elif isinstance(ll, Pair):
        if ind == 0:
            return ll.get_car()
        raise SchemeEvaluationError

def concatenate(lists):
    for ll in lists:
        if not is_list(ll):
            raise SchemeEvaluationError
    def helper(list_ind):
        if list_ind == len(lists):
            return "nil"
        cur_list = lists[list_ind]
        def subhelper(i):
            if i == list_len([cur_list]):
                return helper(list_ind+1)
            return Pair(find_ind([cur_list, i]), subhelper(i+1))
        return subhelper(0)
    return helper(0)

def map_fn(inp):
    if len(inp) != 2:
        raise SchemeEvaluationError
    fn, ll = inp[0], inp[1]
    len_list = list_len([ll])
    if (not isinstance(fn, Function) and not callable(fn)) or not is_list(ll):
        raise SchemeEvaluationError
    return concatenate([Pair(eval_fn(fn, [find_ind([ll, i])]), "nil")
                        for i in range(len_list)])

def filter_fn(inp):
    if len(inp) != 2:
        raise SchemeEvaluationError
    fn, ll = inp[0], inp[1]
    len_list = list_len([ll])
    if (not isinstance(fn, Function) and not callable(fn)) or not is_list(ll):
        raise SchemeEvaluationError
    return concatenate([Pair(find_ind([ll, i]), "nil") for i in range(len_list)
                        if eval_fn(fn, [find_ind([ll, i])]) is True])

def reduce(inp):
    if len(inp) != 3:
        raise SchemeEvaluationError
    fn, ll, init = inp[0], inp[1], inp[2]
    if ((not isinstance(fn, Function) and not callable(fn)) or not is_list(ll)):
        raise SchemeEvaluationError
    ans = init
    for ind in range(list_len([ll])):
        val = find_ind([ll, ind])
        ans = eval_fn(fn, [ans, val])
    return ans

scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": mult,
    "/": lambda args: args[0] / mult(args[1:]),
    "not": not_fn,
    "#t": True,
    "#f": False,
    "equal?": lambda args: all(args[i-1] == args[i] for i in range(1, len(args))),
    "<": lambda args: all(args[i-1] < args[i] for i in range(1, len(args))),
    "<=": lambda args: all(args[i-1] <= args[i] for i in range(1, len(args))),
    ">": lambda args: all(args[i-1] > args[i] for i in range(1, len(args))),
    ">=": lambda args: all(args[i-1] >= args[i] for i in range(1, len(args))),
    "car": lambda pair: retrieve_cons(pair)[0],
    "cdr": lambda pair: retrieve_cons(pair)[1],
    "nil": "nil",
    "list?": check_list,
    "length": list_len,
    "list-ref": find_ind,
    "append": concatenate,
    "map": map_fn,
    "filter": filter_fn,
    "reduce": reduce,
    "begin": lambda args: args[-1]
}


##############
# Evaluation #
##############

def evaluate_file(filename, frame=None):
    if frame is None:
        frame = {"parent": scheme_builtins}
    with open(filename) as f:
        data = f.read()
    return evaluate(parse(tokenize(data)), frame)

def find_in_frame(tree, frame):
    cur_frame = frame
    while tree not in cur_frame and "parent" in cur_frame.keys():
        cur_frame = cur_frame["parent"]
    if tree in cur_frame:
        return cur_frame, cur_frame[tree]
    raise SchemeNameError


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse Function
    """
    if frame is None:
        frame = {"parent": scheme_builtins}
    if not isinstance(tree, list):
        x = number_or_symbol(tree)
        if isinstance(x, str):
            val = find_in_frame(tree, frame)[1]
            return val
        else:
            if float(x) == float(tree):
                return x
            return float(tree)
    elif len(tree) == 0:
        raise SchemeEvaluationError
    elif tree[0] == "if":
        if evaluate(tree[1], frame) is True:
            return evaluate(tree[2], frame)
        return evaluate(tree[3], frame)
    elif tree[0] == "and":
        for i in range(1, len(tree)):
            if evaluate(tree[i], frame) is False:
                return False
        return True
    elif tree[0] == "or":
        for i in range(1, len(tree)):
            if evaluate(tree[i], frame) is True:
                return True
        return False
    elif tree[0] == "cons":
        if len(tree) != 3:
            raise SchemeEvaluationError
        return Pair(evaluate(tree[1], frame), evaluate(tree[2], frame))
    elif tree[0] == "list":
        def convert_list(ind):
            if ind == len(tree):
                return "nil"
            return Pair(evaluate(tree[ind], frame), convert_list(ind+1))
        return convert_list(1)
    elif tree[0] == "define":
        if isinstance(tree[1], list): #shorthand
            frame[tree[1][0]] = evaluate(["lambda", tree[1][1:len(tree[1])],
                                          tree[2]], frame)
            return frame[tree[1][0]]
        else:
            frame[tree[1]] = evaluate(tree[2], frame)
            return frame[tree[1]]
    elif tree[0] == "lambda":
        return Function(tree[1], tree[2], frame)
    elif tree[0] == "del":
        var = tree[1]
        if var not in frame:
            raise SchemeNameError
        val = frame[var]
        frame.pop(var)
        return val
    elif tree[0] == "let":
        new_frame = {"parent": frame}
        for binding in tree[1]:
            var, val = binding[0], binding[1]
            new_frame[var] = evaluate(val, frame)
        return evaluate(tree[2], new_frame)
    elif tree[0] == "set!":
        var, expr = tree[1], tree[2]
        val = evaluate(expr, frame)
        new_frame = find_in_frame(var, frame)[0]
        new_frame[var] = val
        return val
    else: # should be function call
        fn = evaluate(tree[0], frame)
        args = [evaluate(tree[i], frame) for i in range(1, len(tree))]
        return eval_fn(fn, args)

def eval_fn(fn, args):
    if isinstance(fn, Function):
        new_frame = {"parent": fn.get_frame()}
        ind = 0
        if len(args) > len(fn.get_params()):
            raise SchemeEvaluationError
        for arg in fn.get_params():
            if ind >= len(args):
                raise SchemeEvaluationError
            new_frame[arg] = args[ind]
            ind += 1
        return evaluate(fn.get_body(), new_frame)
    elif not callable(fn):
        raise SchemeEvaluationError
    else: # builtin
        return fn(args)

def result_and_frame(tree, frame=None):
    if frame is None:
        frame = {"parent": scheme_builtins}
    return evaluate(tree, frame), frame

class Function:
    """
    A representation of a function, with a list of parameters, a body, and
    an enclosing frame.
    """
    def __init__(self, params, body, frame):
        self.params = params
        self.body = body
        self.frame = frame
    def get_params(self):
        return self.params
    def get_body(self):
        return self.body
    def get_frame(self):
        return self.frame

class Pair:
    """
    """
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr
    def get_car(self):
        return self.car
    def get_cdr(self):
        return self.cdr
    def set_car(self, val):
        self.car = val
    def set_cdr(self, val):
        self.cdr = val

########
# REPL #
########

import os
import re
import sys
import traceback
from cmd import Cmd

try:
    import readline
except:
    readline = None


def supports_color():
    """
    Returns True if the running system"s terminal supports color, and False
    otherwise.  Not guaranteed to work in all cases, but maybe in most?
    """
    plat = sys.platform
    supported_platform = plat != "Pocket PC" and (
        plat != "win32" or "ANSICON" in os.environ
    )
    # IDLE does not support colors
    if "idlelib" in sys.modules:
        return False
    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True


class SchemeREPL(Cmd):
    """
    Class that implements a Read-Evaluate-Print Loop for our Scheme
    interpreter.
    """

    history_file = os.path.join(os.path.expanduser("~"), ".6101_scheme_history")

    if supports_color():
        prompt = "\033[96min>\033[0m "
        value_msg = "  out> \033[92m\033[1m%r\033[0m"
        error_msg = "  \033[91mEXCEPTION!! %s\033[0m"
    else:
        prompt = "in> "
        value_msg = "  out> %r"
        error_msg = "  EXCEPTION!! %s"

    keywords = {
        "define", "lambda", "if", "equal?", "<", "<=", ">", ">=", "and", "or",
        "del", "let", "set!", "+", "-", "*", "/", "#t", "#f", "not", "nil",
        "cons", "list", "cat", "cdr", "list-ref", "length", "append", "begin",
    }

    def __init__(self, use_frames=False, verbose=False):
        self.verbose = verbose
        self.use_frames = use_frames
        self.global_frame = None
        Cmd.__init__(self)

    def preloop(self):
        if readline and os.path.isfile(self.history_file):
            readline.read_history_file(self.history_file)

    def postloop(self):
        if readline:
            readline.set_history_length(10_000)
            readline.write_history_file(self.history_file)

    def completedefault(self, text, line, begidx, endidx):
        try:
            bound_vars = set(self.global_frame)
        except:
            bound_vars = set()
        return sorted(i for i in (self.keywords | bound_vars) if i.startswith(text))

    def onecmd(self, line):
        if line in {"EOF", "quit", "QUIT"}:
            print()
            print("bye bye!")
            return True

        elif not line.strip():
            return False

        try:
            token_list = tokenize(line)
            if self.verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if self.verbose:
                print("expression>", expression)
            if self.use_frames:
                output, self.global_frame = result_and_frame(
                    *(
                        (expression, self.global_frame)
                        if self.global_frame is not None
                        else (expression,)
                    )
                )
            else:
                output = evaluate(expression)
            print(self.value_msg % output)
        except SchemeError as e:
            if self.verbose:
                traceback.print_tb(e.__traceback__)
                print(self.error_msg.replace("%s", "%r") % e)
            else:
                print(self.error_msg % e)

        return False

    completenames = completedefault

    def cmdloop(self, intro=None):
        while True:
            try:
                Cmd.cmdloop(self, intro=None)
                break
            except KeyboardInterrupt:
                print("^C")


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    import os
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import schemerepl
    my_frame = {"parent": scheme_builtins}
    # print(sys.argv)
    # for x in sys.argv:
    #     if x != "lab.py":
    #         evaluate_file(x, my_frame)
    schemerepl.SchemeREPL(use_frames=True, global_frame = my_frame,
                          verbose=False).cmdloop()
