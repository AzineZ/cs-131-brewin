from intbase import InterpreterBase, ErrorType
from brewparse import parse_program

class LazyWrapper:
    def __init__(self, expr, evaluator):
        """
        Initialize with the expression and a function that evaluates it.
        :param expr: The expression to evaluate lazily.
        :param evaluator: A function that performs the evaluation.
        """
        self.expr = expr
        self.evaluator = evaluator
        self.value = None
        self.evaluated = False

    def get_value(self):
        if not self.evaluated:
            self.value = self.evaluator(self.expr)
            self.evaluated = True
        return self.value

class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)

        self.funcs = {} # {(name,n_args):element,}
        self.vars = [] # [({name:val,},bool),]
        self.bops = {'+', '-', '*', '/', '==', '!=', '>', '>=', '<', '<=', '||', '&&'}

    def run(self, program):
        ast = parse_program(program)

        for func in ast.get('functions'):
            self.funcs[(func.get('name'),len(func.get('args')))] = func

        main_key = None

        for k in self.funcs:
            if k[0] == 'main':
                main_key = k
                break

        if main_key is None:
            super().error(ErrorType.NAME_ERROR, '')
        
        try:
            self.run_fcall(self.funcs[main_key], standalone=True)
        except RuntimeError as e:
            super().error(ErrorType.FAULT_ERROR, f"Uncaught exception: {e}")

    def run_vardef(self, statement):
        name = statement.get('name')

        if name in self.vars[-1][0]:
            super().error(ErrorType.NAME_ERROR, f"Variable {name} is already defined")

        self.vars[-1][0][name] = None

    def run_assign(self, statement):
        name = statement.get('name')
        expr = statement.get('expression')
        
        for scope_vars, is_func in self.vars[::-1]:
            if name in scope_vars:
                captured_vars = {k: v for k, v in scope_vars.items()}  
                scope_vars[name] = LazyWrapper(expr, lambda e: self.run_expr_with_scope(e, captured_vars))
                return

            if is_func: break

        super().error(ErrorType.NAME_ERROR, '')
    
    def run_expr_with_scope(self, expr, captured_vars):
        # Temporarily replace self.vars with the captured scope
        original_vars = self.vars
        self.vars = [(captured_vars, False)]
        try:
            return self.run_expr(expr)
        finally:
            self.vars = original_vars

    def run_fcall(self, statement, standalone=False):
        fcall_name, args = statement.get('name'), statement.get('args')

        if fcall_name == 'inputi' or fcall_name == 'inputs':
            if len(args) > 1:
                super().error(ErrorType.NAME_ERROR, '')

            if args:
                super().output(str(self.run_expr(args[0])))

            res = super().get_input()

            return int(res) if fcall_name == 'inputi' else res

        if fcall_name == 'print':
            out = ''

            for arg in args:
                c_out = self.run_expr(arg)
                if type(c_out) == bool:
                    out += str(c_out).lower()
                else:
                    out += str(c_out)

            super().output(out)

            return None
        
        if (fcall_name, len(args)) not in self.funcs:
            super().error(ErrorType.NAME_ERROR, '')

        func_def = self.funcs[(fcall_name, len(args))]

        template_args = [a.get('name') for a in func_def.get('args')]

        captured_scope = {k: v for scope_vars, _ in self.vars for k, v in scope_vars.items()}
        passed_args = [LazyWrapper(arg, lambda e: self.run_expr_with_scope(e, captured_scope)) for arg in args]

        try:
            self.vars.append(({k: v for k, v in zip(template_args, passed_args)}, True))
            res, _ = self.run_statements(func_def.get('statements'))
            if standalone == False:
                return res.get_value() if isinstance(res, LazyWrapper) else res
            return
        except RuntimeError as e:
            raise  # Propagate exception

        finally:
            # Ensure scope is cleaned up
            self.vars.pop()


    def run_if(self, statement):
        cond = self.run_expr(statement.get('condition'))

        if type(cond) != bool:
            super().error(ErrorType.TYPE_ERROR, '')

        self.vars.append(({}, False))

        res, ret = None, False

        if cond:
            res, ret = self.run_statements(statement.get('statements'))
        elif statement.get('else_statements'):
            res, ret = self.run_statements(statement.get('else_statements'))

        self.vars.pop()

        return res, ret

    def run_for(self, statement):
        res, ret = None, False

        self.run_assign(statement.get('init'))

        while True:
            cond = self.run_expr(statement.get('condition'))

            if type(cond) != bool:
                super().error(ErrorType.TYPE_ERROR, '')

            if ret or not cond: break

            self.vars.append(({}, False))
            res, ret = self.run_statements(statement.get('statements'))
            self.vars.pop()

            self.run_assign(statement.get('update'))

        return res, ret

    def run_return(self, statement):
        expr = statement.get('expression')
        if expr:
            # return self.run_expr(expr)
            captured_scope = {k: v for scope_vars, _ in self.vars for k, v in scope_vars.items()}  # Flatten the current scope
            return LazyWrapper(expr, lambda e: self.run_expr_with_scope(e, captured_scope))
        return None

    def run_statements(self, statements):
        res, ret = None, False

        for statement in statements:
            kind = statement.elem_type

            if kind == 'vardef':
                self.run_vardef(statement)
            elif kind == '=':
                self.run_assign(statement)
            elif kind == 'fcall':
                self.run_fcall(statement, standalone=True)
            elif kind == 'if':
                res, ret = self.run_if(statement)
                if ret: break
            elif kind == 'for':
                res, ret = self.run_for(statement)
                if ret: break
            elif kind == 'return':
                res = self.run_return(statement)
                ret = True
                break
            elif kind == 'raise':
                self.run_raise(statement)
            elif kind == 'try':
                try:
                    res, ret = self.run_try(statement)
                    if ret: break
                except RuntimeError as e:
                    raise

        return res, ret
    
    def run_raise(self, statement):
        exception_type = self.run_expr(statement.get('exception_type'))
        if not isinstance(exception_type, str):
            super().error(ErrorType.TYPE_ERROR, "Raised exception must be a string.")
        raise RuntimeError(exception_type)
    
    def run_try(self, statement):
        try:
            self.vars.append(({}, False))  # Create a new variable scope for the try block
            res, ret = self.run_statements(statement.get('statements'))
            self.vars.pop()
            if ret:
                return res, True  # Exit try/catch on return
        except RuntimeError as e:
            self.vars.pop()  # Pop the try block scope
            exception_type = str(e)
            for catcher in statement.get('catchers'):
                if catcher.get('exception_type') == exception_type:
                    self.vars.append(({}, False))  # New scope for the catch block
                    res, ret = self.run_statements(catcher.get('statements'))
                    self.vars.pop()
                    return res, ret
            # No matching catch block; propagate exception
            raise
        return None, False

    def run_expr(self, expr):
        kind = expr.elem_type

        if kind == 'int' or kind == 'string' or kind == 'bool':
            return expr.get('val')

        elif kind == 'var':
            var_name = expr.get('name')

            for scope_vars, is_func in self.vars[::-1]:
                if var_name in scope_vars:
                    # return scope_vars[var_name]
                    var_value = scope_vars[var_name]

                    # If the variable is a LazyWrapper, evaluate and return its value
                    if isinstance(var_value, LazyWrapper):
                        return var_value.get_value()
                    return var_value

                if is_func: break

            super().error(ErrorType.NAME_ERROR, '')

        elif kind == 'fcall':
            return self.run_fcall(expr)

        elif kind in self.bops:
            l = self.run_expr(expr.get('op1'))
            tl = type(l)
            # Short circuit if left side is bool and op is logical AND or OR
            if tl == bool and (kind == '&&' or kind == '||'):
                if l is True and kind == '||' or l is False and kind == '&&':
                    return l
            

            r = self.run_expr(expr.get('op2'))
            tr = type(r)

            if kind == '==': return tl == tr and l == r
            if kind == '!=': return not (tl == tr and l == r)

            if tl == str and tr == str:
                if kind == '+': return l + r

            if tl == int and tr == int:
                if kind == '+': return l + r
                if kind == '-': return l - r
                if kind == '*': return l * r
                #if kind == '/': return l // r
                if kind == '<': return l < r
                if kind == '<=': return l <= r
                if kind == '>': return l > r
                if kind == '>=': return l >= r
                # add division, checks for 0 in denominator
                if kind == '/':
                    if r == 0:
                        raise RuntimeError("div0")
                    return l // r
            
            if tl == bool and tr == bool:
                if kind == '&&': return l and r
                if kind == '||': return l or r

            super().error(ErrorType.TYPE_ERROR, '')

        elif kind == 'neg':
            o = self.run_expr(expr.get('op1'))
            if type(o) == int: return -o
            
            super().error(ErrorType.TYPE_ERROR, '')

        elif kind == '!':
            o = self.run_expr(expr.get('op1'))
            if type(o) == bool: return not o

            super().error(ErrorType.TYPE_ERROR, '')

        return None

# def main():  # COMMENT THIS ONCE FINISH TESTINGd
#     program = """
# func main() {
#   var a;
#   foo("entered function");
# }

# func foo(a) {
#   print(a);
#   var a;
# }
#             """

#     interpreter = Interpreter()
#     interpreter.run(program)

# main()