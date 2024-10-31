from intbase import InterpreterBase, ErrorType
from brewparse import parse_program


class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)   # call InterpreterBase's constructor

        self.var_to_val = {}
        self.non_var_value_type = {'int', 'string', 'bool'}

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
            super().error(ErrorType.TYPE_ERROR, "Incompatible types for arithmetic operation")
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

    def do_definition(self, statement_node):
        var_name = statement_node.get('name')
        if var_name not in self.var_to_val:
            # We dont need to worry about initial value for Project 1
            self.var_to_val[var_name] = '# Not Initialized #'
            return
        # If the variable already exists, raise error
        self.report_error(var_name, "var_defined")

    def do_assignment(self, statement_node):
        var_name = statement_node.get('name')
        # Check if variable exists
        if var_name not in self.var_to_val:
            self.report_error(var_name, "var_not_defined")
        result = self.do_expression(statement_node.get('expression'))
        self.var_to_val[var_name] = result

    def do_func_call(self, statement_node, origin):
        func_name = statement_node.get('name')
        # Check for valid function names
        if func_name not in ['print', 'inputi']:
            self.report_error(func_name, "func_not_defined")
        # Check for valid function calls
        if func_name == 'print' and origin == 'expression' or func_name == 'inputi' and origin == 'statement':
            self.report_error(func_name, "invalid_func_call")

        func_args = statement_node.get('args')
        if func_name == 'print':
            self.do_print(func_args)
        elif func_name == 'inputi':
            return self.do_inputi(func_args)

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
                if var_name not in self.var_to_val:
                    self.report_error(var_name, "var_not_defined")
                value = self.var_to_val[var_name]
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

    def do_expression(self, arg):
        arg_type = arg.elem_type
        if arg_type in self.non_var_value_type:
            return arg.get('val')
        
        elif arg_type == 'var':
            var_name = arg.get('name')
            # Check if variable exists
            if var_name not in self.var_to_val:
                self.report_error(var_name, "var_not_defined")
            return self.var_to_val[var_name]
        
        elif arg_type == 'neg':
            op1 = self.do_expression(arg.get('op1'))
            return -op1
        
        elif arg_type == '+' or arg_type == '-' or arg_type == '*' or arg_type == '/':
            op1 = self.do_expression(arg.get('op1'))
            op2 = self.do_expression(arg.get('op2'))

            # Check if both operands are of the same type
            if type(op1) != type(op2):                              #Check if we need to check for string type for concat
                self.report_error(None, "mismatched_type")

            if arg_type == '+':
                return op1 + op2
            elif arg_type == '-':
                return op1 - op2
            elif arg_type == '*':
                return op1 * op2
            elif arg_type == '/':
                return op1 // op2
            #return op1 + op2 if arg_type == '+' else op1 - op2
        elif arg_type == 'fcall':
            return self.do_func_call(arg, 'expression')


def main():  # COMMENT THIS ONCE FINISH TESTING
    program = """func main() {
             var x;
             var y;
             y = -2;
             x = inputi("Enter a number: ");
             print("The sum is: ", y - x);
          }"""

    interpreter = Interpreter()
    interpreter.run(program)

main()

# x = inputi("Enter a number: ");