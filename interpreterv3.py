from intbase import InterpreterBase, ErrorType
from brewparse import parse_program
from env_v1 import EnvironmentManager

class ReturnValue:
    def __init__(self, value=None):
        self.value = value

class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)   # call InterpreterBase's constructor

        self.env_stack = []
        self.func_table = {}
        #ADD STRUCT LATERRRRRR and default value of nil for structs!!!!
        self.valid_types = {'int', 'string', 'bool'}
        self.default_values = {'int': 0, 'string': "", 'bool': False}

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
            "var_not_defined": f"Variable {item} does not exist or is out of scope!",
            "func_not_defined": f"The {item} function has not been correctly defined or you passed the wrong number of arguments!",
            "invalid_func_call": f"The {item} function is called in an invalid manner!",
            "invalid_params_to_inputi": "No inputi() function found that takes > 1 parameter"
        }
        if error_type == "mismatched_type":
            super().error(ErrorType.TYPE_ERROR, f"Incompatible types for the {item} operation")
        elif error_type == "invalid_if_condition":
            super().error(ErrorType.TYPE_ERROR, f"Invalid if or for condition. It must be a boolean!")
        elif error_type == "invalid_params_or_ret_type":
            super().error(ErrorType.TYPE_ERROR, f"Invalid parameters or return types for function {item}()!")
        elif error_type == "invalid_var_def":
            super().error(ErrorType.TYPE_ERROR, f"Type of variable {item} is invalid or missing!")
        elif error_type == "Invalid_type_assignment":
            super().error(ErrorType.TYPE_ERROR, f"Incompatible type assignment for variable {item}")
        else:
            super().error(ErrorType.NAME_ERROR, error_messages.get(error_type))

    def check_var_in_env_stack(self, var_name):
        for i in range(len(self.env_stack) - 1, -1, -1):
            if var_name in self.env_stack[i].environment:
                return self.env_stack[i]

            # Stop searching when we reach a function-level environment
            if getattr(self.env_stack[i], 'is_function_scope', False):
                break
    
        return self.report_error(var_name, "var_not_defined")

    def create_env(self, is_func_scope=False):
        new_env = EnvironmentManager()
        #dynamically add an attribute to the object
        new_env.is_function_scope = is_func_scope  # Add a marker to indicate function-level scope
        self.env_stack.append(new_env)
    
    def pop_env(self):
        self.env_stack.pop()

    def get_main_function_node(self, ast):
        funcs = ast.get('functions')
        for func in funcs:
            if func.get('name') == 'main':
                return func
        self.report_error(None, "main_not_found")

    def set_func_table(self, ast):
        funcs = ast.get('functions')
        for func in funcs:
            ret_type = func.get('return_type')
            if ret_type not in self.valid_types and ret_type != 'void':
                self.report_error(func.get('name'), 'invalid_params_or_ret_type')

            args = func.get('args')
            for arg in args:
                if arg.get('var_type') not in self.valid_types:
                    self.report_error(func.get('name'), 'invalid_params_or_ret_type')
            self.func_table[(func.get('name'), len(func.get('args')))] = func
        #print(self.func_table)
        
    def run(self, program):
        ast = parse_program(program)
        self.set_func_table(ast)
        self.run_func(self.get_main_function_node(ast))
        self.env_stack = []
        self.func_table = {}
    
    # this is where we execute the body of a {} scope
    def run_func(self, func_node, args=None):
        self.create_env(is_func_scope=True)

        # Assign arguments to function parameters and ensure copying
        params = func_node.get('args')
        if args:
            for param_node, arg_value in zip(params, args):
                param_name = param_node.get('name')
                self.env_stack[-1].create(param_name, arg_value)

        for statement_node in func_node.get('statements'):
            result = self.run_statement(statement_node)
            if isinstance(result, ReturnValue):  # Check for an explicit return
                self.pop_env()
                return result.value  # Extract the actual value and return it

        self.pop_env()
        return None  # Return nil if no explicit return is found

    def run_statement(self, statement_node):
        statement_type = statement_node.elem_type
        if statement_type == 'vardef':
            self.do_definition(statement_node)
        elif statement_type == '=':
            self.do_assignment(statement_node)
        elif statement_type == 'fcall':
            return self.do_func_call(statement_node, 'statement')
        elif statement_type == 'if':
            return self.do_if(statement_node)
        elif statement_type == 'for':
            return self.do_for(statement_node)
        elif statement_type == 'return':
            return self.do_return(statement_node)
    
    def do_return(self, statement_node):
        exp = statement_node.get('expression')
        if exp is not None:
            return ReturnValue(self.do_expression(exp))  # Return an instance with the evaluated expression
        return ReturnValue(value=None)  # Return an instance with None to indicate an implicit return
    
    # Need to add supports for struct-to-struct, struct-to-nil, and coercion!!!!
    def do_type_comp(self, lhs, rhs):
        if type(lhs) != type(rhs):
            return False
        return True
    
    def run_block(self, statement_node):
        for statement_node in statement_node.get('statements'):
            result = self.run_statement(statement_node)
            if isinstance(result, ReturnValue):  # Check for an explicit return
                return result  # Propagate the ReturnValue object up the chain
    
    def do_for(self, statement_node):
        #Initialize the counter
        starter = statement_node.get('init')
        self.run_statement(starter)

        #Apply condition, check if it's boolean
        condition = self.do_expression(statement_node.get('condition'))
        condition = self.do_coercion(condition)  # Coerce to bool if needed
        update = statement_node.get('update')

        if not isinstance(condition, bool):
            self.report_error(None, "invalid_if_condition")
        
        #Main loop function. If condition is correct, run its statement, then run update statement and repeat
        while condition:
            self.create_env()
            result = self.run_block(statement_node)
            if isinstance(result, ReturnValue):  # Check for an explicit return
                self.pop_env()
                return result
            
            self.pop_env()
            self.run_statement(update)
            condition = self.do_expression(statement_node.get('condition'))        
    
    # this is a {} block
    def do_if(self, statement_node):
        condition = self.do_expression(statement_node.get('condition'))
        condition = self.do_coercion(condition)  # Coerce to bool if needed

        if not isinstance(condition, bool):
            self.report_error(None, "invalid_if_condition")
            #return

        self.create_env()
        if condition:
            result = self.run_block(statement_node)
            self.pop_env()
            if isinstance(result, ReturnValue):  # Propagate the ReturnValue object
                return result
        else:
            self.pop_env()
            return self.do_else(statement_node)
    
    def do_else(self, else_statements):
        e_statements = else_statements.get('else_statements')
        if e_statements:
            self.create_env()
            for statement_node in e_statements:
                result = self.run_statement(statement_node)
                if isinstance(result, ReturnValue):  # Check for an explicit return
                    self.pop_env()
                    return result
            self.pop_env()


    def do_definition(self, statement_node):
        var_name, var_type = statement_node.get('name'), statement_node.get('var_type')
        curr_env = self.env_stack[-1]
        if var_type not in self.valid_types:
            self.report_error(var_name, "invalid_var_def")
        #check for existing variable in the scope
        if not curr_env.get(var_name):
            curr_env.create(var_name, self.default_values[var_type])
            return

        # If the variable already exists, raise error
        self.report_error(var_name, "var_defined")

    def do_assignment(self, statement_node):
        var_name = statement_node.get('name')
        # Check if variable exists and return appropriate environment
        curr_env = self.check_var_in_env_stack(var_name)
        result = self.do_expression(statement_node.get('expression'))
        existing_value = curr_env.get(var_name)
        
        if self.do_type_comp(existing_value, result) is False:
            #Check for coercion if the two are of different types
            if isinstance(existing_value, bool) and isinstance(result, int):
                result = self.do_coercion(result)
            else:
                self.report_error(var_name, "Invalid_type_assignment")
        curr_env.set(var_name, result)

    def do_func_call(self, statement_node, origin):
        func_name = statement_node.get('name')
        func_args = statement_node.get('args')

        if func_name not in ['print', 'inputi', 'inputs']:
            # Check if the function exists and get the matching function node
            func_node = self.func_table.get((func_name, len(func_args)))
            if not func_node:
                self.report_error(func_name, "func_not_defined")

            # Evaluate arguments and pass them as a copy to the function
            evaluated_args = [self.copy_value(self.do_expression(arg)) for arg in func_args]
            result = self.run_func(func_node, evaluated_args)
            return result if result is not None else None  # Ensure consistent nil handling

        # Handle built-in functions as before
        if func_name == 'print':
            self.do_print(func_args)
            if origin == 'expression':
                return None  # Return nil if print is called from an expression
        elif func_name == 'inputi':
            return self.do_inputi(func_args)
        elif func_name == 'inputs':
            return self.do_inputs(func_args)
    
    def copy_value(self, value):
        x = value
        return x

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
                curr_env = self.check_var_in_env_stack(var_name)
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

    def do_coercion(self, value):
        if isinstance(value, int):
            return value != 0
        return value
    
    def do_expression(self, arg):
        arg_type = arg.elem_type
        if arg_type in self.non_var_value_type:
            return arg.get('val')
        
        elif arg_type == 'var':
            var_name = arg.get('name')
            # Check if variable exists and return the correct environment
            curr_env = self.check_var_in_env_stack(var_name)
            return curr_env.get(var_name)
        
        elif arg_type == 'neg':
            op1 = self.do_expression(arg.get('op1'))
            # Need to do two separate checks because bool is a subset of int
            if not isinstance(op1, int) or isinstance(op1, bool):
                self.report_error(arg_type, "mismatched_type")
            return -op1
        
        elif arg_type == '!':
            op1 = self.do_expression(arg.get('op1'))
            # Coerce int to bool if necessary
            op1 = self.do_coercion(op1)
            if not isinstance(op1, bool):
                self.report_error(arg_type, "mismatched_type")
            return not op1
        
        elif arg_type in self.int_ops or arg_type in self.string_ops or arg_type in self.bool_ops or arg_type in self.nil_ops:
            op1 = self.do_expression(arg.get('op1'))
            op2 = self.do_expression(arg.get('op2'))

            # Handle coercion for comparisons (e.g., ==, !=)
            if arg_type in ['==', '!=', '&&', '||']:
                op1 = self.do_expression(arg.get('op1'))
                op2 = self.do_expression(arg.get('op2'))

                if arg_type in ['&&', '||']:
                    # Logical operators: both operands should be coerced to bool
                    op1 = self.do_coercion(op1)
                    op2 = self.do_coercion(op2)

                    # Ensure both operands are boolean after coercion
                    if not isinstance(op1, bool) or not isinstance(op2, bool):
                        self.report_error(arg_type, "mismatched_type")
                    return op1 and op2 if arg_type == '&&' else op1 or op2

                elif arg_type in ['==', '!=']:
                    # Equality comparisons: coerce int to bool if one operand is bool
                    if isinstance(op1, int) and isinstance(op2, bool):
                        op1 = self.do_coercion(op1)
                    elif isinstance(op2, int) and isinstance(op1, bool):
                        op2 = self.do_coercion(op2)

                    # Perform the equality comparison
                    return (op1 == op2) if arg_type == '==' else (op1 != op2)

            # Check if both operands are of the same type or coercible
            if type(op1) != type(op2):
                if arg_type in ['==', '!=']:
                    return (op1 == op2) if arg_type == '==' else (op1 != op2)
                self.report_error(arg_type, "mismatched_type")
            
            # Check binary arg_type with operands of the same type
            operation = None
            if isinstance(op1, str) and isinstance(op2, str) and arg_type in self.string_ops:
                operation = self.string_ops.get(arg_type)
            elif isinstance(op1, int) and not isinstance(op1, bool) and isinstance(op2, int) and not isinstance(op2, bool) and arg_type in self.int_ops:
                operation = self.int_ops.get(arg_type)
            elif isinstance(op1, bool) and isinstance(op2, bool) and arg_type in self.bool_ops:
                operation = self.bool_ops.get(arg_type)
            elif op1 == None and op2 == None and arg_type in self.nil_ops:
                operation = self.nil_ops.get(arg_type)
            
            if not operation:
                self.report_error(arg_type, "mismatched_type")
            else:
                return operation(op1, op2)

        elif arg_type == 'fcall':
            return self.do_func_call(arg, 'expression')


def main():  # COMMENT THIS ONCE FINISH TESTING
    program = """
             func main() : int {
                var k: int;
                for (k = 5; k; k = k - 1) {
                    print(-1 && true);
                }      
            }
            """

    interpreter = Interpreter()
    interpreter.run(program)

main()