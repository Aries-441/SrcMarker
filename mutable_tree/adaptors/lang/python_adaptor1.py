# https://github.com/tree-sitter/tree-sitter-python/blob/master/grammar.js
import tree_sitter
from ...nodes import Expression, Statement, node_factory
from ...nodes.statements.statement_list import StatementList
from ...nodes import (
    ArrayAccess,
    ArrayExpression,
    AssignmentExpression,
    BinaryExpression,
    CallExpression,
    FieldAccess,
    Identifier,
    Literal,
    UnaryExpression,
    ParenthesizedExpression,
    ExpressionList,
    FormalParameter,
)
from ...nodes import (
    BlockStatement,
    BreakStatement,
    ContinueStatement,
    EmptyStatement,
    ExpressionStatement,
    ForInStatement,
    ForStatement,
    IfStatement,
    ReturnStatement,
    WhileStatement,
    TryStatement,
    CatchClause,
    WithStatement,
)
# 导入 Python 特有的语句类型
from ...nodes.statements.pass_stmt import PassStatement
from ...nodes.statements.assert_stmt import AssertStatement
from ...nodes.statements.raise_stmt import RaiseStatement
from ...nodes.statements.with_statement import WithStatement
from ...nodes import (
    FunctionDeclaration,
    FunctionHeader,
    ClassDeclaration,
    VariableDeclarator,
    VariableDeclaration,
)
from ...nodes import NodeType

import logging

logger = logging.getLogger(__name__)

# 定义字面量类型常量
class LiteralTypes:
    STRING = "string"
    NUMBER = "number"
    NULL = "null"
    BOOLEAN = "boolean"

# 定义布尔运算符常量
class BooleanOps:
    AND = "and"
    OR = "or"


def convert_program(node: tree_sitter.Node) -> BlockStatement:
    """
    将Python程序转换为内部表示的BlockStatement
    """
    statements = []
    for child in node.children:
        if child.type == "_statement":
            stmt = convert_statement(child)
            if stmt is not None:
                statements.append(stmt)
    
    # 创建 StatementList 对象
    statement_list = node_factory.create_statement_list(statements)
    return node_factory.create_block_stmt(statement_list)

def convert_expression(node: tree_sitter.Node) -> Expression:
    """
    将Python表达式转换为内部表示的Expression
    """
    if node is None:
        return None
    
    expr_convertors = {
        "identifier": convert_identifier,
        "string": convert_literal,
        "integer": convert_literal,
        "float": convert_literal,
        "true": convert_literal,
        "false": convert_literal,
        "none": convert_literal,
        "binary_operator": convert_binary_operator,
        "boolean_operator": convert_boolean_operator,
        "call": convert_call,
        "attribute": convert_attribute,
        "subscript": convert_subscript,
        "list": convert_list,
        "tuple": convert_tuple,
        "dictionary": convert_dictionary,
        "set": convert_set,
        "parenthesized_expression": convert_parenthesized_expression,
        "assignment": convert_assignment,
        "augmented_assignment": convert_augmented_assignment,
        "unary_operator": convert_unary_operator,
        "comparison_operator": convert_comparison_operator,
        "expression_list": convert_expression_list,
    }
    
    if node.type in expr_convertors:
        return expr_convertors[node.type](node)
    
    # 处理表达式
    if node.type == "expression":
        for child in node.children:
            expr = convert_expression(child)
            if expr is not None:
                return expr
    
    logger.warning(f"未处理的表达式类型: {node.type}")
    return None


def convert_statement(node: tree_sitter.Node) -> Statement:
    """
    将Python语句转换为内部表示的Statement
    """
    stmt_convertors = {
        "function_definition": convert_function_definition,
        "expression_statement": convert_expression_statement,
        "return_statement": convert_return_statement,
        "if_statement": convert_if_statement,
        "for_statement": convert_for_statement,
        "while_statement": convert_while_statement,
        "pass_statement": convert_pass_statement, 
        "break_statement": lambda n: node_factory.create_break_stmt(),
        "continue_statement": lambda n: node_factory.create_continue_stmt(),
        "assert_statement": convert_assert_statement,
        "raise_statement": convert_raise_statement, 
        "with_statement": convert_with_statement,
        "try_statement": convert_try_statement,
        "class_definition": convert_class_definition,
    }

    if node.type in stmt_convertors:
        return stmt_convertors[node.type](node)
    
    # 处理简单语句列表
    if node.type == "_simple_statements":
        statements = []
        for child in node.children:
            if child.type == "_simple_statement":
                stmt = convert_statement(child)
                if stmt is not None:
                    statements.append(stmt)
        statement_list = node_factory.create_statement_list(statements)
        return node_factory.create_block_stmt(statement_list)
    
    # 处理复合语句
    if node.type == "_compound_statement":
        for child in node.children:
            stmt = convert_statement(child)
            if stmt is not None:
                return stmt
    
    # 处理块语句
    if node.type == "block":
        statements = []
        for child in node.children:
            if child.type == "_statement":
                stmt = convert_statement(child)
                if stmt is not None:
                    statements.append(stmt)
        statement_list = node_factory.create_statement_list(statements)
        return node_factory.create_block_stmt(statement_list)
    
    logger.warning(f"未处理的语句类型: {node.type}")
    return None

#NOTE: 需要确认NodeType.CLASS_DECLARATION是否存在，如果不存在需要添加
def convert_class_definition(node: tree_sitter.Node) -> ClassDeclaration:
    """
    将Python类定义转换为内部表示的ClassDeclaration
    """
    name_node = node.child_by_field_name("name")
    body_node = node.child_by_field_name("body")
    
    name = convert_expression(name_node) if name_node else None
    body = convert_statement(body_node) if body_node else None
    
    # 处理继承
    bases = []
    arguments_node = node.child_by_field_name("superclasses")
    if arguments_node:
        for child in arguments_node.children:
            if child.type == "expression":
                base = convert_expression(child)
                if base is not None:
                    bases.append(base)
    
    return node_factory.create_class_declaration(name, bases, body)


def convert_parameter(node: tree_sitter.Node) -> FormalParameter:
    """
    将Python参数转换为内部表示的FormalParameter
    """
    if node.type == "identifier":
        # 简单参数，如 def func(x):
        identifier = convert_expression(node)
        decl = node_factory.create_variable_declarator(identifier)
        return node_factory.create_untyped_param(decl)
    
    if node.type == "typed_parameter":
        # 类型注解参数，如 def func(x: int):
        name_node = node.child_by_field_name("name")
        name = convert_expression(name_node) if name_node else None
        decl = node_factory.create_variable_declarator(name)
        return node_factory.create_untyped_param(decl)
    
    if node.type == "default_parameter":
        # 默认值参数，如 def func(x=10):
        name_node = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")
        
        name = convert_expression(name_node) if name_node else None
        value = convert_expression(value_node) if value_node else None
        
        decl = node_factory.create_variable_declarator(name)
        if value is not None:
            decl = node_factory.create_initializing_declarator(decl, value)
        return node_factory.create_untyped_param(decl)
    
    # 处理 *args 和 **kwargs 类型的参数
    if node.type == "list_splat_pattern" or node.type == "dictionary_splat_pattern":
        child_node = node.child(0)
        if child_node and child_node.type == "identifier":
            identifier = convert_expression(child_node)
            decl = node_factory.create_variable_declarator(identifier)
            return node_factory.create_spread_param(decl)
    
    logger.warning(f"未处理的参数类型: {node.type}")
    return None

def convert_function_definition(node: tree_sitter.Node) -> FunctionDeclaration:
    """
    将Python函数定义转换为内部表示的FunctionDeclaration
    """
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    body_node = node.child_by_field_name("body")
    
    # 获取函数名标识符
    name = convert_expression(name_node) if name_node else None
    name_decl = node_factory.create_variable_declarator(name)
    
    # 处理参数
    parameters = []
    if params_node and params_node.child_count > 2:  # 跳过括号
        for i in range(1, params_node.child_count - 1):
            param_node = params_node.children[i]
            if param_node.type == ",":
                continue
            param = convert_parameter(param_node)
            if param is not None:
                parameters.append(param)
    
    # 创建参数列表
    param_list = node_factory.create_formal_parameter_list(parameters)
    
    # 创建函数声明器
    func_decl = node_factory.create_func_declarator(name_decl, param_list)
    
    # 创建函数头
    func_header = node_factory.create_func_header(func_decl)
    
    # 处理函数体
    body_stmts = []
    if body_node:
        for stmt_node in body_node.children:
                stmt = convert_statement(stmt_node)
                if stmt is not None:
                    body_stmts.append(stmt)
    
    # 如果函数体为空，添加一个pass语句
    if not body_stmts:
        body_stmts.append(node_factory.create_pass_stmt())
    
    body = node_factory.create_block_stmt(node_factory.create_statement_list(body_stmts))
    
    # 创建并返回函数声明
    return node_factory.create_func_declaration(func_header, body)


def convert_expression_statement(node: tree_sitter.Node) -> ExpressionStatement:
    expr = convert_expression(node.children[0])
    return node_factory.create_expression_stmt(expr)


def convert_return_statement(node: tree_sitter.Node) -> ReturnStatement:
    """
    将Python返回语句转换为内部表示的ReturnStatement
    """
    expr_node = None
    for child in node.children:
        if child.type in ["expression", "expression_list"]:
            expr_node = child
            break
    
    expr = convert_expression(expr_node) if expr_node else None
    return node_factory.create_return_stmt(expr)


def convert_if_statement(node: tree_sitter.Node) -> IfStatement:
    """
    将Python if语句转换为内部表示的IfStatement
    """
    condition_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    alternative_node = node.child_by_field_name("alternative")
    
    condition = convert_expression(condition_node) if condition_node else None
    consequence = convert_statement(consequence_node) if consequence_node else None
    alternative = convert_statement(alternative_node) if alternative_node else None
    
    return node_factory.create_if_stmt(condition, consequence, alternative)


def convert_for_statement(node: tree_sitter.Node) -> ForInStatement:
    """
    将Python for语句转换为内部表示的ForInStatement
    """
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")
    body_node = node.child_by_field_name("body")
    
    left = convert_expression(left_node) if left_node else None
    right = convert_expression(right_node) if right_node else None
    body = convert_statement(body_node) if body_node else None
    
    return node_factory.create_for_in_stmt(left, right, body)


def convert_while_statement(node: tree_sitter.Node) -> WhileStatement:
    """
    将Python while语句转换为内部表示的WhileStatement
    """
    condition_node = node.child_by_field_name("condition")
    body_node = node.child_by_field_name("body")
    
    condition = convert_expression(condition_node) if condition_node else None
    body = convert_statement(body_node) if body_node else None
    
    return node_factory.create_while_stmt(condition, body)

#NOTE: 需要确认NodeType.ASSERT_STMT是否存在，如果不存在需要添加
def convert_assert_statement(node: tree_sitter.Node) -> AssertStatement:
    """
    将Python assert语句转换为内部表示的AssertStatement
    """
    expressions = []
    for child in node.children:
        if child.type == "expression":
            expr = convert_expression(child)
            if expr is not None:
                expressions.append(expr)
    
    condition = expressions[0] if expressions else None
    message = expressions[1] if len(expressions) > 1 else None
    
    return AssertStatement(NodeType.ASSERT_STMT, condition, message)

#NOTE: 需要确认NodeType.PASS_STMT是否存在，如果不存在需要添加
def convert_pass_statement(node: tree_sitter.Node) -> PassStatement:
    """
    将Python pass语句转换为内部表示的PassStatement
    """
    return node_factory.create_pass_stmt()

#NOTE: 需要确认NodeType.RAISE_STMT是否存在，如果不存在需要添加
def convert_raise_statement(node: tree_sitter.Node) -> RaiseStatement:
    """
    将Python raise语句转换为内部表示的RaiseStatement
    """
    expression = None
    cause = None
    
    for child in node.children:
        if child.type == "expression":
            if expression is None:
                expression = convert_expression(child)
            else:
                # 如果已经有expression，那么这个是cause (from 子句)
                cause = convert_expression(child)
    
    return RaiseStatement(NodeType.RAISE_STMT, expression, cause)


def convert_with_statement(node: tree_sitter.Node) -> WithStatement:
    """
    将Python with语句转换为内部表示的WithStatement
    """
    object_node = None
    body_node = node.child_by_field_name("body")
    
    # 查找with项
    for child in node.children:
        if child.type == "with_item":
            object_node = child.child_by_field_name("value")
            break
    
    object_expr = convert_expression(object_node) if object_node else None
    body = convert_statement(body_node) if body_node else None
    
    return node_factory.create_with_stmt(object_expr, body)


def convert_try_statement(node: tree_sitter.Node) -> TryStatement:
    """
    将Python try语句转换为内部表示的TryStatement
    """
    body_node = node.child_by_field_name("body")
    
    body = convert_statement(body_node) if body_node else None
    catch_clauses = []
    finally_block = None
    
    for child in node.children:
        if child.type == "catch_clause":
            catch_clauses.append(convert_catch_clause(child))
        elif child.type == "finally_clause":
            finally_block = convert_statement(child)

    return TryStatement(NodeType.TRY_STMT, body, catch_clauses, finally_block)

def convert_dictionary(node: tree_sitter.Node) -> Expression:
    """
    将Python字典转换为内部表示的Expression
    """
    elements = []
    for child in node.children:
        if child.type == "pair":
            key_node = child.child_by_field_name("key")
            value_node = child.child_by_field_name("value")
            
            key = convert_expression(key_node) if key_node else None
            value = convert_expression(value_node) if value_node else None
            
            if key is not None and value is not None:
                # 将键值对表示为二元表达式
                elements.append(BinaryExpression(NodeType.BINARY_EXPR, key, value, ":"))
    
    expr_list = ExpressionList(NodeType.EXPRESSION_LIST, elements)
    return ArrayExpression(NodeType.ARRAY_EXPR, expr_list)


def convert_set(node: tree_sitter.Node) -> ArrayExpression:
    """
    将Python集合转换为内部表示的ArrayExpression
    """
    elements = []
    for child in node.children:
        if child.type in ["expression", "list_splat"]:
            element = convert_expression(child)
            if element is not None:
                elements.append(element)
    
    # 使用ArrayExpression表示集合，可以在stringifier中特殊处理
    expr_list = ExpressionList(NodeType.EXPRESSION_LIST, elements)
    return ArrayExpression(NodeType.ARRAY_EXPR, expr_list)


def convert_parenthesized_expression(node: tree_sitter.Node) -> ParenthesizedExpression:
    """
    将Python括号表达式转换为内部表示的ParenthesizedExpression
    """
    for child in node.children:
        if child.type != "(" and child.type != ")":
            expr = convert_expression(child)
            if expr is not None:
                return ParenthesizedExpression(NodeType.PARENTHESIZED_EXPR, expr)
    
    return None


def convert_assignment(node: tree_sitter.Node) -> AssignmentExpression:
    """
    将Python赋值表达式转换为内部表示的AssignmentExpression
    """
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")
    
    left = convert_expression(left_node) if left_node else None
    right = convert_expression(right_node) if right_node else None
    
    return AssignmentExpression(NodeType.ASSIGNMENT_EXPR, left, right, "=")


def convert_augmented_assignment(node: tree_sitter.Node) -> AssignmentExpression:
    """
    将Python增强赋值表达式转换为内部表示的AssignmentExpression
    """
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")
    operator_node = node.child_by_field_name("operator")
    
    left = convert_expression(left_node) if left_node else None
    right = convert_expression(right_node) if right_node else None
    operator = operator_node.text.decode('utf-8') if operator_node else "+="
    
    return AssignmentExpression(NodeType.ASSIGNMENT_EXPR, left, right, operator)


def convert_unary_operator(node: tree_sitter.Node) -> UnaryExpression:
    """
    将Python一元运算符转换为内部表示的UnaryExpression
    """
    argument_node = node.child_by_field_name("argument")
    operator_node = node.child_by_field_name("operator")
    
    argument = convert_expression(argument_node) if argument_node else None
    operator = operator_node.text.decode('utf-8') if operator_node else "-"
    
    return UnaryExpression(NodeType.UNARY_EXPR, argument, operator)


def convert_comparison_operator(node: tree_sitter.Node) -> BinaryExpression:
    """
    将Python比较运算符转换为内部表示的BinaryExpression
    """
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")
    operator_node = node.child_by_field_name("operator")
    
    left = convert_expression(left_node) if left_node else None
    right = convert_expression(right_node) if right_node else None
    operator = operator_node.text.decode('utf-8') if operator_node else "=="
    
    return BinaryExpression(NodeType.BINARY_EXPR, left, right, operator)


def convert_try_statement(node: tree_sitter.Node) -> TryStatement:
    """
    将Python try语句转换为内部表示的TryStatement
    """
    body_node = node.child_by_field_name("body")
    
    body = convert_statement(body_node) if body_node else None
    catch_clauses = []
    finally_block = None
    
    for child in node.children:
        if child.type in ["except_clause", "except_group_clause"]:
            value_node = child.child_by_field_name("value")
            alias_node = child.child_by_field_name("alias")
            
            value = convert_expression(value_node) if value_node else None
            alias = convert_expression(alias_node) if alias_node else None
            
            # 获取except块的主体
            except_body = None
            for except_child in child.children:
                if except_child.type == "_suite":
                    except_body = convert_statement(except_child)
                    break
            
            catch_clause = CatchClause(NodeType.CATCH_CLAUSE, except_body, value, alias)
            catch_clauses.append(catch_clause)
        
        elif child.type == "finally_clause":
            for finally_child in child.children:
                if finally_child.type == "_suite":
                    finally_block = convert_statement(finally_child)
                    break
    
    return TryStatement(NodeType.TRY_STMT, body, catch_clauses, finally_block)


def convert_return_statement(node: tree_sitter.Node) -> ReturnStatement:
    """
    将Python返回语句转换为内部表示的ReturnStatement
    """
    expr_node = None
    for child in node.children:
        if child.type in ["expression", "expression_list"]:
            expr_node = child
            break
    
    expr = convert_expression(expr_node) if expr_node else None
    return ReturnStatement(NodeType.RETURN_STMT, expr)


# 修复字面量转换函数
def convert_string(node: tree_sitter.Node) -> Literal:
    content = node.text.decode('utf-8')
    return Literal(content, NodeType.STRING)

def convert_integer(node: tree_sitter.Node) -> Literal:
    return Literal(node.text.decode('utf-8'), NodeType.NUMBER)

def convert_float(node: tree_sitter.Node) -> Literal:
    return Literal(node.text.decode('utf-8'), NodeType.NUMBER)

def convert_none(node: tree_sitter.Node) -> Literal:
    return Literal("None", NodeType.NULL)

def convert_boolean(node: tree_sitter.Node) -> Literal:
    value = "True" if node.type == "true" else "False"
    return Literal(value, NodeType.BOOLEAN)


# 使用node_factory创建节点的函数
def convert_expression_statement(node: tree_sitter.Node) -> ExpressionStatement:
    expr = convert_expression(node.children[0])
    return node_factory.create_expression_stmt(expr)


# 修复其他函数中的NodeType参数
def convert_literal(node: tree_sitter.Node) -> Literal:
    """
    将Python字面量转换为内部表示的Literal
    """
    if node.type == "string":
        return convert_string(node)
    elif node.type == "integer":
        return convert_integer(node)
    elif node.type == "float":
        return convert_float(node)
    elif node.type == "true" or node.type == "false":
        return convert_boolean(node)
    elif node.type == "none":
        return convert_none(node)
    else:
        # 默认处理
        value = node.text.decode('utf-8')
        return Literal(value, NodeType.STRING)