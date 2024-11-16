from intbase import InterpreterBase, ErrorType
from brewparse import parse_program
from env_v1 import EnvironmentManager

class ReturnValue:
    def __init__(self, value=None):
        self.value = value
    
    def has_value(self):
        """Check if the return value is actually a value (not NoValue)."""
        return not isinstance(self.value, NoValue)

class NoValue:
    """Marker class to indicate a return with no value (used for void functions)."""
    def __repr__(self):
        return "<NoValue>"

class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)   # call InterpreterBase's constructor

        self.env_stack = []
        self.func_table = {}
        self.struct_table = {}
        self.var_to_struct_type = {}
        self.var_to_prims = {}
        self.valid_types = {'int', 'string', 'bool', 'str'}
        #self.default_values = {'int': 0, 'string': "", 'bool': False}

        # contain only primitives
        self.non_var_value_type = {'int', 'string', 'bool', 'str'}
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
        self.struct_ops =  {
            '==': lambda x, y: x is y,
            '!=': lambda x, y: x is not y
        }

    def report_error(self, item, error_type):
        error_map = {
            "mismatched_type": (ErrorType.TYPE_ERROR, f"Incompatible types for the {item} operation"),
            "invalid_if_condition": (ErrorType.TYPE_ERROR, "Invalid if or for condition. It must be a boolean!"),
            "invalid_params_or_ret_type": (ErrorType.TYPE_ERROR, f"Invalid parameters or return types for function {item}()!"),
            "invalid_var_def": (ErrorType.TYPE_ERROR, f"Type of variable {item} is invalid or missing!"),
            "invalid_type_assignment": (ErrorType.TYPE_ERROR, f"Incompatible type assignment for variable {item}"),
            "invalid_void_return": (ErrorType.TYPE_ERROR, f"Void function {item} must not return a value!"),
            "mismatched_return_value": (ErrorType.TYPE_ERROR, f"Function {item} is returning a value of mismatched type!"),
            "void_func_in_expression": (ErrorType.TYPE_ERROR, f"Void function {item} cannot be compared!"),
            "mismatched_param": (ErrorType.TYPE_ERROR, f"Function {item} receives a mismatched type of argument!"),
            "invalid_struct_type": (ErrorType.TYPE_ERROR, f"Struct {item} is not defined!"),
            "nil_struct_access": (ErrorType.FAULT_ERROR, "Nil does not have fields!"),
            "invalid_struct_access": (ErrorType.TYPE_ERROR, f"Struct {item} does not exist!"),
            "invalid_field_name": (ErrorType.NAME_ERROR, f"Field {item} does not exist!"),
            "print_in_expr": (ErrorType.TYPE_ERROR, "Print function must not be used in expression!"),
            "main_not_found": (ErrorType.NAME_ERROR, "No main() function was found"),
            "var_defined": (ErrorType.NAME_ERROR, f"Variable {item} defined more than once!"),
            "var_not_defined": (ErrorType.NAME_ERROR, f"Variable {item} does not exist or is out of scope!"),
            "func_not_defined": (ErrorType.NAME_ERROR, f"Function {item} not defined or wrong number of arguments!"),
            "invalid_func_call": (ErrorType.NAME_ERROR, f"Function {item} called invalidly!"),
            "invalid_params_to_inputi": (ErrorType.NAME_ERROR, "No inputi() function found that takes > 1 parameter")
        }
        
        error_type, message = error_map.get(error_type, (ErrorType.NAME_ERROR, f"Unknown error: {error_type}"))
        super().error(error_type, message)

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

    # Two new errors not implemented yet
    def set_func_table(self, ast):
        structs = ast.get('structs')
        for struct in structs:
            struct_name = struct.get('name')
            # No need to check for duplicate struct names
            self.struct_table[struct_name] = struct

            # Validate struct fields
            fields = struct.get('fields')
            for field in fields:
                field_type = field.get('var_type')
                if field_type not in self.valid_types and field_type not in self.struct_table:
                    self.report_error(struct_name, 'invalid_field_type')

        funcs = ast.get('functions')
        for func in funcs:
            ret_type = func.get('return_type')
            if ret_type not in self.valid_types and ret_type not in self.struct_table and ret_type != 'void':
                self.report_error(func.get('name'), 'invalid_params_or_ret_type')

            args = func.get('args')
            for arg in args:
                arg_type = arg.get('var_type')
                if arg_type not in self.valid_types and arg_type not in self.struct_table:
                    self.report_error(func.get('name'), 'invalid_params_or_ret_type')
            self.func_table[(func.get('name'), len(func.get('args')))] = func
        #print(self.func_table)
        #print(self.struct_table)
        
    def run(self, program):
        ast = parse_program(program)
        self.set_func_table(ast)
        #print(ast)
        self.run_func(self.get_main_function_node(ast))
        self.env_stack = []
        self.func_table = {}
        self.struct_table = {}
        self.var_to_struct_type = {}
    
    # this is where we execute the body of a {} scope
    def run_func(self, func_node, args=None):
        self.create_env(is_func_scope=True)

        ret_type = func_node.get('return_type')
        # Assign arguments to function parameters and ensure copying
        params = func_node.get('args')
        if args:
            for param_node, arg_value in zip(params, args):
                param_name = param_node.get('name')
                expected_type = param_node.get('var_type')

                if isinstance(arg_value, tuple):    # this arg is None sent from a struct variable
                    struct_type = arg_value[1]
                    if struct_type != expected_type:
                        self.report_error(func_node.get('name'), "mismatched_param")
                    self.env_stack[-1].create(param_name, arg_value[0])
                else:
                    #Coerce only when the formal parameter is a bool
                    if expected_type == 'bool':
                        arg_value = self.do_coercion(arg_value)
                    # Check if the coerced value matches the expected type
                    if not self.match_type(arg_value, expected_type):
                        self.report_error(func_node.get('name'), "mismatched_param")

                    self.env_stack[-1].create(param_name, arg_value)

        for statement_node in func_node.get('statements'):
            result = self.run_statement(statement_node)

            if isinstance(result, ReturnValue):  # Check for an explicit return
                return_value = result.value
                empty_return = isinstance(return_value, NoValue)
                self.pop_env()

                # Handle functions with a void return type
                if ret_type == 'void':
                    if not empty_return:
                        self.report_error(func_node.get('name'), "invalid_void_return")
                    return None  # A void function should return nothing
                
                if empty_return:
                    return self.do_func_return(self.get_default_value(ret_type))
                
                if ret_type == 'bool':
                    return_value = self.do_coercion(return_value)
                # Check if the coerced value matches the function's declared return type
                if not self.match_type(return_value, ret_type):
                    self.report_error(func_node.get('name'), "mismatched_return_value")

                # Handle return by reference for structs and by value for primitives
                return self.do_func_return(return_value)

        self.pop_env()

        # Handle the case where the function ends without an explicit return statement
        if ret_type == 'void':
            return None  # Void function should return nothing
        return self.do_func_return(self.get_default_value(ret_type))  # Return default value if no explicit return is encountered
        #return None  # Return nil if no explicit return is found
    
    def do_func_return(self, value):
        if type(value).__name__ not in self.non_var_value_type:
            return value  # Return by reference for user-defined structs
        return self.copy_value(value)  # Return by value for primitives

    def match_type(self, value, expected_type):
        """Check if the value matches the expected type"""
        if expected_type in self.struct_table:
            # Return True if the value is a struct instance (uses a dictionary) of the expected type
            if isinstance(value, dict) and value.get('type') == expected_type:
                return True
            # Return True if the value is nil and expected type is a struct
            if value is None:
                return True
        elif expected_type == 'int' and isinstance(value, int) and not isinstance(value, bool): #Bool is a subset of Int so we need additional check
            return True
        elif expected_type == 'string' and isinstance(value, str):
            return True
        elif expected_type == 'bool' and isinstance(value, bool) or expected_type == 'bool' and isinstance(value, int):
            return True
        # elif expected_type == 'nil' and value is None:
        #     return True
        elif expected_type == 'void' and value is None:
            return True
        return False

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
        # If there's an expression, evaluate and return its value
        if exp is not None:
            return ReturnValue(self.do_expression(exp))  # Explicitly returning a value (can be None if it's nil)
        # If there's no expression, return NoValue to indicate no value was returned
        return ReturnValue(value=NoValue())
    
    # Need to add supports for struct-to-struct, struct-to-nil!!!!
    def do_type_comp(self, lhs, rhs):
        # Case 1: Handle uninitialized variables (lhs is None)
        if lhs is None:
            # If lhs is None but is meant to be a struct type, allow assignment if rhs is of the correct struct type
            if isinstance(rhs, dict) and 'type' in rhs or rhs is None:
                return True  # Allow assignment if rhs is a valid struct instance
            return False

        # Case 2: If rhs is None (nil), lhs must be a struct type
        if rhs is None:
            return isinstance(lhs, dict) and 'type' in lhs

        # Case 3: Handle assignment for struct types
        if isinstance(lhs, dict) and 'type' in lhs:
            # lhs is a struct type, rhs must be a struct of the same type
            if isinstance(rhs, dict) and 'type' in rhs:
                return lhs['type'] == rhs['type']
            return False  # Structs can only accept other structs of the same type or nil

        # Case 4: Handle assignment for primitive types
        lhs_type = type(lhs).__name__
        rhs_type = type(rhs).__name__

        # Allow coercion from int to bool
        if lhs_type == 'bool' and rhs_type == 'int':
            return True

        # Check if both lhs and rhs are of the same primitive type
        return lhs_type == rhs_type
    
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
        if var_type not in self.valid_types and var_type not in self.struct_table:
            self.report_error(var_name, "invalid_var_def")
        #check for existing variable in the scope
        if curr_env.get(var_name) is None:
            curr_env.create(var_name, self.get_default_value(var_type))    # Struct should be handled here too
            if var_type not in self.valid_types:
                self.var_to_struct_type[var_name] = var_type               # Parse variable with struct type
            else:
                self.var_to_prims[var_name] = var_type
            return

        # If the variable already exists, raise error
        self.report_error(var_name, "var_defined")

    def do_assignment(self, statement_node):
        var_name = statement_node.get('name')
        target_type = self.get_struct_type(statement_node.get('expression'))
        result = self.do_expression(statement_node.get('expression')) # This should already handle nested fields

        # Handle nested field assignment directly within this function
        if '.' in var_name:
            # Resolve the LHS (left-hand side) to get the struct instance and field name
            lhs_instance, lhs_field_name = self.do_field_access(var_name)
            struct_type = lhs_instance.get('type')
            
            # Extract the field definition list from the struct type
            field_defs = self.struct_table[struct_type].get('fields')
            
            # Find the field definition node
            lhs_field_type = None
            for field_def in field_defs:
                if field_def.get('name') == lhs_field_name:
                    lhs_field_type = field_def.get('var_type')
                    break
            
            # Perform type checking for nested fields
            if not lhs_field_type or not self.match_type(result, lhs_field_type):
                self.report_error(lhs_field_name, "invalid_type_assignment")

            # if lhs is a struct, check rhs for struct type or pure nil
            lhs = lhs_instance['fields'][lhs_field_name]
            if isinstance(lhs, dict) and target_type is not None and lhs.get('type') != target_type:
                self.report_error(var_name, "invalid_type_assignment")
            lhs_instance['fields'][lhs_field_name] = result
            return

        curr_env = self.check_var_in_env_stack(var_name)
        existing_value = curr_env.get(var_name)

        # If lhs is a struct
        if var_name in self.var_to_struct_type:
            struct_type = self.var_to_struct_type[var_name]
            if target_type is not None and target_type != struct_type:             # rhs is a value from a struct
                self.report_error(var_name, "invalid_type_assignment")
            # if self.match_type(result, struct_type) is False:                       #rhs is a value
            #     self.report_error(var_name, "invalid_type_assignment")
        
        # Type check for regular variable assignment
        elif self.do_type_comp(existing_value, result) is False:
            if isinstance(existing_value, bool) and isinstance(result, int):
                result = self.do_coercion(result)
            else:
                self.report_error(var_name, "invalid_type_assignment")

        # Assign the value to the variable in the environment
        curr_env.set(var_name, result)

    def do_func_call(self, statement_node, origin):
        func_name = statement_node.get('name')
        func_args = statement_node.get('args')

        if func_name not in ['print', 'inputi', 'inputs']:
            # Check if the function exists and get the matching function node
            func_node = self.func_table.get((func_name, len(func_args)))
            if not func_node:
                self.report_error(func_name, "func_not_defined")
            if origin == 'expression' and func_node.get('return_type') == 'void':
                self.report_error(func_name, "void_func_in_expression")

            # Evaluate arguments and pass them as a copy to the function
            evaluated_args = []
            for arg in func_args:
                arg_name = arg.get('name')
                arg_type = None
                if arg_name in self.var_to_struct_type:
                    arg_type = self.var_to_struct_type[arg_name]

                eval_arg = self.do_expression(arg)
                # Check if the type of the evaluated argument is in the set of valid_types
                if type(eval_arg).__name__ in self.non_var_value_type:
                    # Pass by value for int, string, bool
                    evaluated_args.append(self.copy_value(eval_arg))
                else:
                    # Pass by object reference for structs
                    if arg_type is not None and eval_arg is None:       #Thi 
                        evaluated_args.append((eval_arg, arg_type))
                    else:
                        evaluated_args.append(eval_arg)

            result = self.run_func(func_node, evaluated_args)
            return result if result is not None else None  # Ensure consistent nil handling

        # Handle built-in functions as before
        if func_name == 'print':
            if origin == 'expression':
                self.report_error(None, "print_in_expr")
            self.do_print(func_args)
            # if origin == 'expression':
            #     return None  # Return nil if print is called from an expression
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
            
            else:
                value = self.do_expression(arg)

            if arg_type == 'var':
                var_type = None
                arg_name = arg.get('name')
                if arg_name in self.var_to_prims:
                    var_type = self.var_to_prims[arg_name]
                if var_type == 'bool' and isinstance(value, int) and not isinstance(value, bool):
                    value = self.do_coercion(value)
            # Check if the value is a boolean and convert to lowercase if True or False
            if isinstance(value, bool):
                content += str(value).lower()
            else:
                content += str(value)
        if content == "None":
            super().output("nil")
        else:
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
    
    def get_default_value(self, var_type):
        """Returns the default value based on the type."""
        if var_type == 'int':
            return 0
        elif var_type == 'bool':
            return False
        elif var_type == 'string':
            return ""
        else:  # For other structs
            return None  # Represents 'nil'
        
    def do_new_struct(self, new_node):
        struct_type = new_node.get('var_type')
        # Check if the struct type exists in the dictionary
        if struct_type not in self.struct_table:
            self.report_error(struct_type, 'invalid_struct_type')
        
        struct_node = self.struct_table[struct_type]
        fields = {}
        for field_node in struct_node.get('fields'):
            field_name = field_node.get('name')
            field_type = field_node.get('var_type') # can contain structs
            fields[field_name] = self.get_default_value(field_type)
        return {'type': struct_type, 'fields': fields}
    
    def do_field_access(self, var_name):
        parts = var_name.split('.')
        base_var_name = parts[0]
        field_names = parts[1:]

        # Retrieve the base variable from the environment
        curr_env = self.check_var_in_env_stack(base_var_name)
        struct_instance = curr_env.get(base_var_name)

        # Ensure the base variable is a struct instance and not nil
        if struct_instance is None:
            self.report_error(base_var_name, "nil_struct_access")
        if not isinstance(struct_instance, dict) or 'type' not in struct_instance:
            self.report_error(base_var_name, "invalid_struct_access")

        # Traverse through the fields to reach the target field
        for field_name in field_names[:-1]:
            if field_name not in struct_instance.get('fields'):
                self.report_error(field_name, "invalid_field_name")
            struct_instance = struct_instance['fields'][field_name]

            # If any intermediate field is nil, raise an error
            if struct_instance is None:
                self.report_error(field_name, "nil_struct_access")

        final_field_name = field_names[-1]
        struct_type = struct_instance.get('type')
        struct_fields = self.struct_table[struct_type].get('fields')
        
        # Validate that the final field exists in the struct definition
        if final_field_name not in [field.get('name') for field in struct_fields]:
            self.report_error(final_field_name, "invalid_field_name")

        return struct_instance, final_field_name
    
    def get_field_value(self, var_name):
        """
        Retrieves the value of a nested field (e.g., 'd.companion.age').
        """
        struct_instance, field_name = self.do_field_access(var_name)
        return struct_instance.get('fields').get(field_name)


    def do_expression(self, arg):
        arg_type = arg.elem_type
        if arg_type in self.non_var_value_type:
            return arg.get('val')
        
        elif arg_type == 'new':
            return self.do_new_struct(arg)
        
        elif arg_type == 'var':
            var_name = arg.get('name')
            if '.' in var_name:
                return self.get_field_value(var_name)   # Return the evaluated struct field, possibly nested
            else:
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
        
        elif arg_type in self.int_ops or arg_type in self.string_ops or arg_type in self.bool_ops or arg_type in self.nil_ops or arg_type in self.struct_ops:
            # Evaluate operands once
            op1 = self.do_expression(arg.get('op1'))
            op2 = self.do_expression(arg.get('op2'))

            # Handle logical operators separately
            if arg_type in ['&&', '||']:
                # Coerce both operands to bool if necessary
                op1 = self.do_coercion(op1)
                op2 = self.do_coercion(op2)

                # Ensure both operands are boolean after coercion
                if not isinstance(op1, bool) or not isinstance(op2, bool):
                    self.report_error(arg_type, "mismatched_type")
                return op1 and op2 if arg_type == '&&' else op1 or op2

            elif arg_type in ['==', '!=']:
                # Step 1: Handle cases where one or both operands are variables
                type1 = self.get_struct_type(arg.get('op1'))
                type2 = self.get_struct_type(arg.get('op2'))

                # Step 2: If both are variables with struct types, handle type checking
                if type1 and type2:
                    # If one of them is None (uninitialized), ensure they are of the same struct type
                    if op1 is None or op2 is None:
                        if type1 != type2:
                            self.report_error(arg_type, "mismatched_type")
                        return (op1 is None and op2 is None) if arg_type == '==' else (op1 is not None or op2 is not None)
                # If one is a struct and the other is a pure nil value
                elif type1 or type2:
                    return (op1 is None and op2 is None) if arg_type == '==' else (op1 is not None or op2 is not None)

                # Both are allocated structs, ensure they have the same type
                if isinstance(op1, dict) and isinstance(op2, dict):
                    if op1['type'] != op2['type']:
                        self.report_error(arg_type, "mismatched_type")
                    return (op1 is op2) if arg_type == '==' else (op1 is not op2)

                # Step 3: Handle primitive types and coercion
                if isinstance(op1, int) and isinstance(op2, bool):
                    op1 = self.do_coercion(op1)
                elif isinstance(op2, int) and isinstance(op1, bool):
                    op2 = self.do_coercion(op2)

                # Check if both operands are of the same type for arithmetic and string operations
                if type(op1) != type(op2):
                    self.report_error(arg_type, "mismatched_type")
                # Step 4: Perform the equality or inequality comparison for primitives
                return (op1 == op2) if arg_type == '==' else (op1 != op2)
            
            # Determine the operation based on type and arg_type
            operation = None
            if isinstance(op1, str) and isinstance(op2, str) and arg_type in self.string_ops:
                operation = self.string_ops.get(arg_type)
            elif isinstance(op1, int) and not isinstance(op1, bool) and isinstance(op2, int) and not isinstance(op2, bool) and arg_type in self.int_ops:
                operation = self.int_ops.get(arg_type)
            elif isinstance(op1, bool) and isinstance(op2, bool) and arg_type in self.bool_ops:
                operation = self.bool_ops.get(arg_type)
            elif op1 is None and op2 is None and arg_type in self.nil_ops:
                operation = self.nil_ops.get(arg_type)
            # Check if both operands are structs (dictionaries) and use reference comparison
            elif isinstance(op1, dict) and isinstance(op2, dict) and op1['type'] == op2['type'] and arg_type in self.struct_ops:
                # Compare by reference using 'is'
                operation = self.struct_ops.get(arg_type)

            if not operation:
                self.report_error(arg_type, "mismatched_type")
            else:
                return operation(op1, op2)

        elif arg_type == 'fcall':
            return self.do_func_call(arg, 'expression')
    
    def get_struct_type(self, operand):
        if operand.elem_type == 'var':
            var_name = operand.get('name')
            if var_name in self.var_to_struct_type:
                return self.var_to_struct_type[var_name]
        return None


# def main():  # COMMENT THIS ONCE FINISH TESTING
#     program = """
# struct Puppy {
#     name: string;
#     bark: int;
# }

# struct Cat {
#     name: string;
#     meow: int;
# }
# struct Dog {
#   name : string;
#   alive : bool;
#   age: int;
#   offspring: Puppy;
# }

# func main() : void {
#   print(1 && true);
# }
#             """

#     interpreter = Interpreter()
#     interpreter.run(program)

# main()