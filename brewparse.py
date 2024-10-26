from element import Element
from brewlex import *
from intbase import InterpreterBase
from ply import yacc

# Parsing rules

precedence = (
    ("left", "OR"),
    ("left", "AND"),
    ("left", "GREATER_EQ", "GREATER", "LESS_EQ", "LESS", "EQ", "NOT_EQ"),
    ("left", "PLUS", "MINUS"),
    ("left", "MULTIPLY", "DIVIDE"),
    ("right", "UMINUS", "NOT"),
)


def collapse_items(p, group_index, singleton_index):
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[group_index]
        p[0].append(p[singleton_index])


def p_program(p):
    """program : structs funcs
    | funcs"""
    if len(p) == 2:
        p[0] = Element(InterpreterBase.PROGRAM_NODE, structs=[], functions=p[1])
    else:
        p[0] = Element(InterpreterBase.PROGRAM_NODE, structs=p[1], functions=p[2])

def p_structs(p):
    """structs : structs struct
    | struct"""
    collapse_items(p, 1, 2)  # 2 -> struct 

def p_struct(p):
   "struct : STRUCT NAME LBRACE fields RBRACE"
   p[0] = Element(InterpreterBase.STRUCT_NODE, name=p[2], fields=p[4])

def p_fields(p):
   """fields : fields field
   | field"""
   collapse_items(p, 1, 2)  # 2 -> field

def p_field(p):
  "field : NAME COLON NAME SEMI"  # field_name: type
  p[0] = Element(InterpreterBase.FIELD_DEF_NODE, name=p[1], var_type=p[3])

def p_funcs(p):
    """funcs : funcs func
    | func"""
    collapse_items(p, 1, 2)  # 2 -> func

# Note: the second NAME is the return type, not a function name
def p_func(p):
    """func : FUNC NAME LPAREN formal_args RPAREN COLON NAME LBRACE statements RBRACE
    | FUNC NAME LPAREN RPAREN COLON NAME LBRACE statements RBRACE"""
    if len(p) == 11:  # handle with 1+ formal args
        p[0] = Element(InterpreterBase.FUNC_NODE, name=p[2], args=p[4], return_type = p[7], statements=p[9])
    else:  # handle no formal args
        p[0] = Element(InterpreterBase.FUNC_NODE, name=p[2], args=[], return_type = p[6], statements=p[8])

def p_func2(p):
    """func : FUNC NAME LPAREN formal_args RPAREN LBRACE statements RBRACE
    | FUNC NAME LPAREN RPAREN LBRACE statements RBRACE"""
    if len(p) == 9:  # handle with 1+ formal args
        p[0] = Element(InterpreterBase.FUNC_NODE, name=p[2], args=p[4], return_type = None, statements=p[7])
    else:  # handle no formal args
        p[0] = Element(InterpreterBase.FUNC_NODE, name=p[2], args=[], return_type = None, statements=p[6])

def p_formal_args(p):
    """formal_args : formal_args COMMA formal_arg
    | formal_arg"""
    collapse_items(p, 1, 3)  # 3 -> formal_arg

# Note: the second NAME is the return type, not a function name
def p_formal_arg(p):
    """formal_arg : NAME COLON NAME
    | NAME"""
    if len(p) == 2:
      p[0] = Element(InterpreterBase.ARG_NODE, name=p[1], var_type = None)
    else:
      p[0] = Element(InterpreterBase.ARG_NODE, name=p[1], var_type = p[3])

def p_statements(p):
    """statements : statements statement
    | statement"""
    collapse_items(p, 1, 2)  # 3 -> formal_arg


def p_statement___assign(p):
    "statement : assign SEMI"
    p[0] = p[1]

def p_assign(p):
    "assign : variable_w_dot ASSIGN expression"
    p[0] = Element("=", name=p[1], expression=p[3])

def p_statement___var(p):
    """statement : VAR variable COLON NAME SEMI
    | VAR variable SEMI"""
    if len(p) == 6:
      p[0] = Element(InterpreterBase.VAR_DEF_NODE, name=p[2], var_type=p[4])
    else:
      p[0] = Element(InterpreterBase.VAR_DEF_NODE, name=p[2], var_type=None)

def p_variable(p):
    "variable : NAME"
    p[0] = p[1]

def p_variable_w_dot(p):
    """variable_w_dot : variable_w_dot DOT NAME
    | NAME"""
    if len(p) == 4:
        p[0] = p[1] + "." + p[3]
    else:
        p[0] = p[1]

def p_statement_if(p):
    """statement : IF LPAREN expression RPAREN LBRACE statements RBRACE
    | IF LPAREN expression RPAREN LBRACE statements RBRACE ELSE LBRACE statements RBRACE
    """
    if len(p) == 8:
        p[0] = Element(
            InterpreterBase.IF_NODE,
            condition=p[3],
            statements=p[6],
            else_statements=None,
        )
    else:
        p[0] = Element(
            InterpreterBase.IF_NODE,
            condition=p[3],
            statements=p[6],
            else_statements=p[10],
        )

def p_statement_try(p):
    """statement : TRY LBRACE statements RBRACE catchers"""
    p[0] = Element(InterpreterBase.TRY_NODE, statements=p[3], catchers=p[5])

def p_catches(p):
    """catchers : catchers catch
    | catch"""
    collapse_items(p, 1, 2)

def p_catch(p):
    "catch : CATCH STRING LBRACE statements RBRACE"
    p[0] = Element(InterpreterBase.CATCH_NODE, exception_type=p[2], statements=p[4])

def p_statement_for(p):
    "statement : FOR LPAREN assign SEMI expression SEMI assign RPAREN LBRACE statements RBRACE"
    p[0] = Element(InterpreterBase.FOR_NODE, init=p[3], condition=p[5], update=p[7], statements=p[10])

def p_statement_raise(p):
    "statement : RAISE expression SEMI"
    p[0] = Element(InterpreterBase.RAISE_NODE, exception_type=p[2])

def p_statement_expr(p):
    "statement : expression SEMI"
    p[0] = p[1]


def p_statement_return(p):
    """statement : RETURN expression SEMI
    | RETURN SEMI"""
    if len(p) == 4:
        expr = p[2]
    else:
        expr = None
    p[0] = Element(InterpreterBase.RETURN_NODE, expression=expr)


def p_expression_not(p):
    "expression : NOT expression"
    p[0] = Element(InterpreterBase.NOT_NODE, op1=p[2])


def p_expression_uminus(p):
    "expression : MINUS expression %prec UMINUS"
    p[0] = Element(InterpreterBase.NEG_NODE, op1=p[2])

def p_expression_new(p):
    "expression : NEW NAME"
    p[0] = Element(InterpreterBase.NEW_NODE, var_type=p[2])


def p_arith_expression_binop(p):
    """expression : expression EQ expression
    | expression GREATER expression
    | expression LESS expression
    | expression NOT_EQ expression
    | expression GREATER_EQ expression
    | expression LESS_EQ expression
    | expression PLUS expression
    | expression MINUS expression
    | expression MULTIPLY expression
    | expression DIVIDE expression"""
    p[0] = Element(p[2], op1=p[1], op2=p[3])


def p_expression_group(p):
    "expression : LPAREN expression RPAREN"
    p[0] = p[2]


def p_expression_and_or(p):
    """expression : expression OR expression
    | expression AND expression"""
    p[0] = Element(p[2], op1=p[1], op2=p[3])


def p_expression_number(p):
    "expression : NUMBER"
    p[0] = Element(InterpreterBase.INT_NODE, val=p[1])


def p_expression_bool(p):
    """expression : TRUE
    | FALSE"""
    bool_val = p[1] == InterpreterBase.TRUE_DEF
    p[0] = Element(InterpreterBase.BOOL_NODE, val=bool_val)


def p_expression_nil(p):
    "expression : NIL"
    p[0] = Element(InterpreterBase.NIL_NODE)


def p_expression_string(p):
    "expression : STRING"
    p[0] = Element(InterpreterBase.STRING_NODE, val=p[1])


def p_expression_variable(p):
    "expression : variable_w_dot"
    p[0] = Element(InterpreterBase.VAR_NODE, name=p[1])


def p_func_call(p):
    """expression : NAME LPAREN args RPAREN
    | NAME LPAREN RPAREN"""
    if len(p) == 5:
        p[0] = Element(InterpreterBase.FCALL_NODE, name=p[1], args=p[3])
    else:
        p[0] = Element(InterpreterBase.FCALL_NODE, name=p[1], args=[])


def p_expression_args(p):
    """args : args COMMA expression
    | expression"""
    collapse_items(p, 1, 3)


def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}' on line {p.lineno}")
    else:
        print("Syntax error at EOF")


# exported function
def parse_program(program):
    reset_lineno()
    ast = yacc.parse(program)
    if ast is None:
        raise SyntaxError("Syntax error")
    return ast


# generate our parser
yacc.yacc() # yacc.yacc(debug=True, debuglog=open("parse.log", "w"))
