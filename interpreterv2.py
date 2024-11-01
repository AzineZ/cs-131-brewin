from intbase import InterpreterBase, ErrorType
from brewparse import parse_program
from env_v1 import EnvironmentManager


class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)   # call InterpreterBase's constructor

        #self.var_to_val = {}
        self.env_stack = []

        #Fix this when we implement scoping
        unique_env = EnvironmentManager()
        self.env_stack.append(unique_env)

        self.non_var_value_type = {'int', 'string', 'bool'}
        self.string_ops = {'+': lambda x, y: x + y,
                           '==': lambda x, y: x == y,
                           '!=': lambda x, y: x != y
                        }
        self.int_ops = {'+': lambda x, y: x + y,
                        '-': lambda x, y: x - y,
                        '*': lambda x, y: x * y,
                        '/': lambda x, y: x // y,
                        '==': lambda x, y: x == y,
                        '!=': lambda x, y: x != y,
                        '<': lambda x, y: x < y,
                        '<=': lambda x, y: x <= y,
                        '>': lambda x, y: x > y,
                        '>=': lambda x, y: x >= y
                        }
        self.bool_ops = {'||': lambda x, y: x or y,
                         '&&': lambda x, y: x and y,
                         '==': lambda x, y: x == y,
                         '!=': lambda x, y: x != y,
                        }
        self.nil_ops = {
                        '==': lambda x, y: x == y,
                        '!=': lambda x, y: x != y,
                        }

    def report_error(self, item, error_type):
        error_messages = {
            "main_not_found": "No main() function was found",
            "var_defined": f"Variable {item} defined more than once!",
            "var_not_defined": f"Variable {item} does not exist!",
            "func_not_defined": f"The {item} function has not been defined!",
            "invalid_func_call": f"The {item} function is called in an invalid manner!",
            "invalid_params_to_inputi": "No inputi() function found that takes > 1 parameter"
        }
        if error_type == "mismatched_type":
            super().error(ErrorType.TYPE_ERROR, f"Incompatible types for the {item} operation")
        elif error_type == "invalid_if_condition":
            super().error(ErrorType.TYPE_ERROR, f"Invalid if condition. It must be a boolean!")
        else:
            super().error(ErrorType.NAME_ERROR, error_messages.get(error_type))

    def get_main_function_node(self, ast):
        funcs = ast.get('functions')
        for func in funcs:
            if func.get('name') == 'main':
                return func
        self.report_error(None, "main_not_found")

    def run(self, program):
        ast = parse_program(program)
        self.run_func(self.get_main_function_node(ast))

    def run_func(self, func_node):
        for statement_node in func_node.get('statements'):
            self.run_statement(statement_node)

    def run_statement(self, statement_node):
        statement_type = statement_node.elem_type
        if statement_type == 'vardef':
            self.do_definition(statement_node)
        elif statement_type == '=':
            self.do_assignment(statement_node)
        elif statement_type == 'fcall':
            self.do_func_call(statement_node, 'statement')
        elif statement_type == 'if':
            self.do_if(statement_node)
    
    def do_if(self, statement_node):
        #print(statement_node.get('condition'))
        condition = self.do_expression(statement_node.get('condition'))
        if not isinstance(condition, bool):
            self.report_error(None, "invalid_if_condition")
            return
        if not condition:
            # Use a similar run_func called do_else that run else_statements
            self.do_else(statement_node)
        else:
            # Reuse the run_func function to run statements
            self.run_func(statement_node)
    
    def do_else(self, else_statements):
        e_statements = else_statements.get('else_statements')
        if e_statements:
            for statement_node in e_statements:
                self.run_statement(statement_node)


    def do_definition(self, statement_node):
        var_name = statement_node.get('name')
        curr_env = self.env_stack[-1]

        if not curr_env.get(var_name):
            curr_env.create(var_name, '# Not Initialized #')
            return

        # If the variable already exists, raise error
        self.report_error(var_name, "var_defined")

    def do_assignment(self, statement_node):
        var_name = statement_node.get('name')
        curr_env = self.env_stack[-1]
        # Check if variable exists
        if not curr_env.get(var_name):
        #if var_name not in self.var_to_val:
            self.report_error(var_name, "var_not_defined")
        result = self.do_expression(statement_node.get('expression'))
        curr_env.set(var_name, result)
        #self.var_to_val[var_name] = result

    def do_func_call(self, statement_node, origin):
        func_name = statement_node.get('name')
        # Check for valid function names
        if func_name not in ['print', 'inputi', "inputs"]:
            self.report_error(func_name, "func_not_defined")
        # Check for valid function calls
        if func_name == 'inputi' and origin == 'statement' or func_name == 'inputs' and origin == 'statement':
            self.report_error(func_name, "invalid_func_call")

        func_args = statement_node.get('args')
        if func_name == 'print':
            self.do_print(func_args)
            if origin == 'expression':
                return None                     #Return nil if print is called from an expression
        elif func_name == 'inputi':
            return self.do_inputi(func_args)
        elif func_name == 'inputs':
            return self.do_inputs(func_args)

    def do_print(self, args):
        content = ""
        # arguments for print can be a string, an int, an expression, a variable
        for arg in args:
            arg_type = arg.elem_type
            value = ''
            if arg_type in self.non_var_value_type:
                value = arg.get('val')
            elif arg_type == 'var':
                var_name = arg.get('name')
                curr_env = self.env_stack[-1]

                if not curr_env.get(var_name):
                #if var_name not in self.var_to_val:
                    self.report_error(var_name, "var_not_defined")
                #value = self.var_to_val[var_name]
                value = curr_env.get(var_name)
            else:
                value = self.do_expression(arg)

            # Check if the value is a boolean and convert to lowercase if True or False
            if isinstance(value, bool):
                content += str(value).lower()
            else:
                content += str(value)

        super().output(content)

    def do_inputi(self, arg):
        if arg:
            # We assume that this arg list has only zero or one argument
            if len(arg) > 1:
                self.report_error(None, "invalid_params_to_inputi")
            super().output(arg[0].get('val'))
        user_input = int(super().get_input())
        return user_input
    
    def do_inputs(self, arg):
        if arg:
            # We assume that this arg list has only zero or one argument
            if len(arg) > 1:
                self.report_error(None, "invalid_params_to_inputi")
            super().output(arg[0].get('val'))
        user_input = str(super().get_input())
        return user_input

    def do_expression(self, arg):
        arg_type = arg.elem_type
        if arg_type in self.non_var_value_type:
            return arg.get('val')
        
        elif arg_type == 'var':
            var_name = arg.get('name')
            curr_env = self.env_stack[-1]
            # Check if variable exists
            if not curr_env.get(var_name):
            #if var_name not in self.var_to_val:
                self.report_error(var_name, "var_not_defined")
            return curr_env.get(var_name)
            #return self.var_to_val[var_name]
        
        elif arg_type == 'neg':
            op1 = self.do_expression(arg.get('op1'))
            if not isinstance(op1, int):
                self.report_error(arg_type, "mismatched_type")
            return -op1
        
        elif arg_type == '!':
            op1 = self.do_expression(arg.get('op1'))
            if not isinstance(op1, bool):
                self.report_error(arg_type, "mismatched_type")
            return not op1
        
        elif arg_type in self.int_ops or arg_type in self.string_ops or arg_type in self.bool_ops:
            op1 = self.do_expression(arg.get('op1'))
            op2 = self.do_expression(arg.get('op2'))

            # Check if both operands are of the same type
            if type(op1) != type(op2):
                #Check the cases of == and != with different data types
                if arg_type == '==' or arg_type == '!=':
                    return False if arg_type == '==' else True
                              
                self.report_error(arg_type, "mismatched_type")
            
            # Check binary arg_type with operands of the same type
            operation = None
            if isinstance(op1, str) and isinstance(op2, str) and arg_type in self.string_ops:
                operation = self.string_ops.get(arg_type)
            elif isinstance(op1, int) and isinstance(op2, int) and arg_type in self.int_ops:
                operation = self.int_ops.get(arg_type)
            elif isinstance(op1, bool) and isinstance(op2, bool) and arg_type in self.bool_ops:
                operation = self.bool_ops.get(arg_type)
            elif op1 == None and op2 == None and arg_type in self.nil_ops:
                operation = self.bool_ops.get(arg_type)
            
            if not operation:
                self.report_error(arg_type, "mismatched_type")
            else:
                return operation(op1, op2)

        elif arg_type == 'fcall':
            return self.do_func_call(arg, 'expression')


def main():  # COMMENT THIS ONCE FINISH TESTING
    program = """func main() {
             var x;
             var y;
             x = 15;
             if (x > 5) {
                print(x);
                if (x < 30 && x > 10) {
                    print(3*x);
                }
             }
          }"""

    interpreter = Interpreter()
    interpreter.run(program)

main()