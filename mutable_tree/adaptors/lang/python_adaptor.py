'''
FileName: 
Description: 
Autor: Liujunjie/Aries-441
Github: https://github.com/Aries-441
Date: 2025-03-19 19:33:36
E-mail: sjtu.liu.jj@gmail.com/sjtu.1518228705@sjtu.edu.cn
LastEditTime: 2025-04-07 14:31:37
'''
# https://github.com/tree-sitter/tree-sitter-python/blob/master/grammar.js
import tree_sitter

from mutable_tree.nodes.expressions.binary_expr import BinaryOps
from mutable_tree.nodes.expressions.field_access import FieldAccessOps
from mutable_tree.nodes.expressions.list_comprehension_expr import ComprehensionClause, ComprehensionExpression
from ...nodes import Expression, Statement, node_factory
from ...nodes.statements.statement_list import StatementList
from ...nodes import (
    ArrayAccess,
    ArrayExpression,
    ArrayPatternType,
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
    LambdaExpression,
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
from ...nodes import UnaryOps
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

from ...nodes import (
    get_binary_op, 
    get_assignment_op,
    get_unary_op,
    get_forin_type

)

import logging
from collections import Counter

logger = logging.getLogger(__name__)

# 添加全局计数器来收集未处理的类型
unhandled_types_counter = Counter()
unhandled_instances = {}  # 记录每种未处理类型出现在哪些实例中

def record_unhandled_type(type_name, category, instance_id=None):
    """记录未处理的类型，而不是直接打印警告"""
    key = f"未处理的{category}类型: {type_name}"
    unhandled_types_counter[key] += 1
    
    # 记录实例ID
    if instance_id is not None:
        if key not in unhandled_instances:
            unhandled_instances[key] = set()
        unhandled_instances[key].add(instance_id)

    
class ForInType:
    COLON = ":"
    IN = "in"
    OF = "of"

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
    statement_list = StatementList(NodeType.STATEMENT_LIST, statements)
    return BlockStatement(NodeType.BLOCK_STMT, statement_list)



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
        "list_comprehension": convert_list_comprehension,
        "list_splat": convert_list_splat,       
        "dictionary_splat": convert_dictionary_splat, 
        #NOTE：二者可以通用
        
        "tuple": convert_tuple,
        "dictionary": convert_dictionary,
        "set": convert_set,
        "parenthesized_expression": convert_parenthesized_expression,
        "assignment": convert_assignment,
        "augmented_assignment": convert_augmented_assignment,
        "unary_operator": convert_unary_operator,
        "comparison_operator": convert_comparison_operator,
        "expression_list": convert_expression_list,
        
        "keyword_argument": convert_keyword_argument,

        "not_operator": convert_not_operator,
        
        "conditional_expression": convert_conditional_expression,
        
        
        "typed_parameter": convert_typed_parameter,
        "default_parameter": convert_default_parameter,
        "list_splat_pattern": convert_list_splat_parameter,
        "dictionary_splat_pattern": convert_dict_splat_parameter,
        "pattern_list": convert_pattern_list,
        "expression_list": convert_pattern_list,
        
        "lambda": convert_lambda,
        "as_pattern": convert_as_pattern,
        "as_pattern_target": convert_as_pattern_target,
        "concatenated_string": convert_concatenated_string,
    }
    
    if node.type in expr_convertors:
        return expr_convertors[node.type](node)
    
    # 处理表达式
    if node.type == "expression":
        for child in node.children:
            expr = convert_expression(child)
            if expr is not None:
                return expr
    
    record_unhandled_type(node.type, "表达式")
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
        "break_statement": lambda n: BreakStatement(NodeType.BREAK_STMT),
        "continue_statement": lambda n: ContinueStatement(NodeType.CONTINUE_STMT),
        "assert_statement": convert_assert_statement,
        "raise_statement": convert_raise_statement, 
        "with_statement": convert_with_statement,
        "try_statement": convert_try_statement,
        "class_definition": convert_class_definition,
        
        "elif_clause": convert_elif_clause,
        "else_clause": convert_else_clause,
        
        'block': convert_block_statement,
        
        #import_statement
        #import_from_statement
        #
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
        statement_list = StatementList(NodeType.STATEMENT_LIST, statements)
        return BlockStatement(NodeType.BLOCK_STMT, statement_list)
    
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
        statement_list = StatementList(NodeType.STATEMENT_LIST, statements)
        return BlockStatement(NodeType.BLOCK_STMT, statement_list)
    
    record_unhandled_type(node.type, "语句")
    return None

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
    #NOTE：CLASS_DECLARATION未被定义
    return ClassDeclaration(NodeType.CLASS_DECLARATION, name, body, bases)


def convert_parameter(node: tree_sitter.Node) -> FormalParameter:
    """
    将Python参数转换为内部表示的FormalParameter
    """
    if node is None:
        return None
    
    # 根据参数类型调用相应的处理函数
    param_converters = {
        "identifier": convert_simple_parameter,
        "typed_parameter": convert_typed_parameter,
        "default_parameter": convert_default_parameter,
        "list_splat_pattern": convert_list_splat_parameter,
        "dictionary_splat_pattern": convert_dict_splat_parameter,
        "list_splat": convert_list_splat,
        "dictionary_splat": convert_list_splat, #NOTE：二者可以通用
    }
    
    if node.type in param_converters:
        return param_converters[node.type](node)
    
    record_unhandled_type(node.type, "参数")
    return None

def convert_simple_parameter(node: tree_sitter.Node) -> FormalParameter:
    """处理简单参数，如 def func(x):"""
    identifier = convert_expression(node)
    decl = node_factory.create_variable_declarator(identifier)
    return node_factory.create_untyped_param(decl)

def convert_typed_parameter(node: tree_sitter.Node) -> FormalParameter:
    """处理类型注解参数，如 def func(x: int):"""
    name_node = node.child_by_field_name("name")
    name = convert_expression(name_node) if name_node else None
    decl = node_factory.create_variable_declarator(name)
    return node_factory.create_untyped_param(decl)

def convert_default_parameter(node: tree_sitter.Node) -> FormalParameter:
    """处理默认值参数，如 def func(x=10):"""
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")
    
    name = convert_expression(name_node) if name_node else None
    value = convert_expression(value_node) if value_node else None
    
    decl = node_factory.create_variable_declarator(name)
    if value is not None:
        decl = node_factory.create_initializing_declarator(decl, value)
    return node_factory.create_untyped_param(decl)

def convert_list_splat_parameter(node: tree_sitter.Node) -> FormalParameter:
    """处理列表展开参数，如 def func(*args):"""
    # 获取参数名（通常是第二个子节点，第一个是*符号）
    child_node = node.child(1)
    if child_node and child_node.type == "identifier":
        identifier = convert_expression(child_node)
        decl = node_factory.create_variable_declarator(identifier)
        
        # 创建一个默认的 DeclaratorType
        type_id = node_factory.create_type_identifier("any")
        decl_type = node_factory.create_declarator_type(type_id)
        
        # 创建展开参数，并标记为非字典展开
        param = node_factory.create_spread_param(decl, decl_type)
        param.is_dict_spread = False
        return param
    
    #logger.warning(f"无法处理的列表展开参数: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
    return None

def convert_dict_splat_parameter(node: tree_sitter.Node) -> FormalParameter:
    """处理字典展开参数，如 def func(**kwargs):"""
    # 获取参数名（通常是第二个子节点，第一个是**符号）
    child_node = node.child(1)
    if child_node and child_node.type == "identifier":
        identifier = convert_expression(child_node)
        decl = node_factory.create_variable_declarator(identifier)
        
        # 创建一个默认的 DeclaratorType
        type_id = node_factory.create_type_identifier("any")
        decl_type = node_factory.create_declarator_type(type_id)
        
        # 创建展开参数，并标记为字典展开
        param = node_factory.create_spread_param(decl, decl_type)
        param.is_dict_spread = True
        return param
    
    #logger.warning(f"无法处理的字典展开参数: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
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
    

    #body = convert_statement(node.child_by_field_name("body"))
    # 创建并返回函数声明
    return node_factory.create_func_declaration(func_header, body)


def convert_expression_statement(node: tree_sitter.Node) -> ExpressionStatement:
    expr = convert_expression(node.children[0])
    return node_factory.create_expression_stmt(expr)


def convert_return_statement(node: tree_sitter.Node) -> Statement:
    """
    处理return语句
    """
    if node.child_count == 1:
        # 处理空return语句
        return node_factory.create_return_stmt()

    expr_node = node.children[1]
    
    # 检查是否是多值返回（逗号分隔的表达式列表）
    if expr_node.type == "expression_list":
        # 创建一个元组表达式来包装多个返回值
        expressions = []
        for child in expr_node.children:
            if child.type != ",":  # 跳过逗号
                expr = convert_expression(child)
                if expr:
                    expressions.append(expr)
        
        # 创建一个数组表达式来表示元组
        expressions = ExpressionList(NodeType.EXPRESSION_LIST, expressions)
        pattern_expr = node_factory.create_array_expr(expressions, ArrayPatternType.PATTERN)
        return node_factory.create_return_stmt(pattern_expr)
    else:
        # 单值返回
        expr = convert_expression(expr_node)
        return node_factory.create_return_stmt(expr)


def convert_if_statement(node: tree_sitter.Node) -> IfStatement:
    """
    将Python if语句转换为内部表示的IfStatement
    """
    cond_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    
    # 获取所有alternative
    alternative_nodes = []
    for child in node.children:
        if child.type == "elif_clause" or child.type == "else_clause":
            alternative_nodes.append(child)
    
    condition = convert_expression(cond_node)
    consequence = convert_statement(consequence_node)
    
    # 处理alternative节点
    alternative = None
    if alternative_nodes:
        # 从最后一个alternative节点开始，构建嵌套的if语句结构
        for alt_node in reversed(alternative_nodes):
            if alt_node.type == "else_clause":
                # 处理else子句
                else_body_node = alt_node.child_by_field_name("body")
                if else_body_node:
                    alternative = convert_statement(else_body_node)
                else:
                    for child in alt_node.children:
                        if child.type != "else" and child.type != ":":
                            alternative = convert_statement(child)
                            break
            elif alt_node.type == "elif_clause":
                # 处理elif子句，将其转换为嵌套的if语句
                elif_cond_node = alt_node.child_by_field_name("condition")
                elif_consequence_node = alt_node.child_by_field_name("consequence")
                
                elif_condition = convert_expression(elif_cond_node)
                elif_consequence = convert_statement(elif_consequence_node)
                
                # 创建一个新的if语句，其alternative是之前处理的alternative
                alternative = node_factory.create_if_stmt(elif_condition, elif_consequence, alternative)

    return node_factory.create_if_stmt(condition, consequence, alternative)
#FIXME:此次需要修改
def convert_for_statement(node: tree_sitter.Node) -> ForInStatement:
    """
    将Python for语句转换为内部表示的ForInStatement
    确保循环体中的语句被正确处理
    """
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")
    body_node = node.child_by_field_name("body")
    
    # 获取迭代变量和可迭代对象
    left = convert_expression(left_node) if left_node else None
    right = convert_expression(right_node) if right_node else None
    
    # 处理循环体，确保所有语句都被正确转换
    body_stmts = []
    if body_node:
        for stmt_node in body_node.children:
            stmt = convert_statement(stmt_node)
            if stmt is not None:
                body_stmts.append(stmt)
    
    # 如果循环体为空，添加一个pass语句
    if not body_stmts:
        body_stmts.append(PassStatement(NodeType.PASS_STMT))
    
    # 创建循环体块
    body = BlockStatement(NodeType.BLOCK_STMT, StatementList(NodeType.STATEMENT_LIST, body_stmts))
    
    # 创建一个表示"let"类型的TypeIdentifier
    type_identifier = node_factory.create_type_identifier("let")
    
    # 使用TypeIdentifier创建DeclaratorType
    decl_type = node_factory.create_declarator_type(type_identifier)
    
    # 使用left作为声明器
    declarator = left
    
    # Python的for循环使用"in"关键字
    forin_type = ForInType.IN
    
    # 检查是否为异步for循环
    is_async = False
    for child in node.children:
        if child.type == "async":
            is_async = True
            break
    
    # 确保right不为空，提供合理的默认值
    if right is None:
        #logger.warning(f"for循环中缺少可迭代对象: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
        # 创建一个空列表作为默认的可迭代对象
        expr_list = ExpressionList(NodeType.EXPRESSION_LIST, [])
        right = ArrayExpression(NodeType.ARRAY_EXPR, expr_list)
    
    return ForInStatement(
        NodeType.FOR_IN_STMT,
        decl_type,
        declarator,
        right,
        body,
        forin_type,
        is_async
    )

def convert_while_statement(node: tree_sitter.Node) -> WhileStatement:
    """
    将Python while语句转换为内部表示的WhileStatement
    """
    condition_node = node.child_by_field_name("condition")
    body_node = node.child_by_field_name("body")
    
    condition = convert_expression(condition_node) if condition_node else None
    body = convert_statement(body_node) if body_node else None
    
    return WhileStatement(NodeType.WHILE_STMT, condition, body)


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
    
    # 使用 NodeType.ASSERT_STMT 作为参数
    return AssertStatement(NodeType.ASSERT_STMT, condition, message)


def convert_pass_statement(node: tree_sitter.Node) -> PassStatement:
    """
    将Python pass语句转换为内部表示的PassStatement
    """
    return PassStatement(NodeType.PASS_STMT)

def convert_raise_statement(node: tree_sitter.Node) -> RaiseStatement:
    """
    将Python raise语句转换为内部表示的RaiseStatement
    
    处理以下形式的raise语句:
    1. raise - 不带参数的重新抛出异常
    2. raise Exception - 抛出指定异常
    3. raise Exception("message") - 带消息的异常
    4. raise Exception from cause - 带原因的异常
    """
    expression = None
    cause = None
    
    # 遍历子节点查找表达式和cause
    for i, child in enumerate(node.children):
        if child.type == "identifier" or child.type == "call":
            # 第一个表达式是异常对象
            if expression is None:
                expression = convert_expression(child)
        elif child.type == "expression":
            # 第一个表达式是异常对象
            if expression is None:
                expression = convert_expression(child)
            # 第二个表达式是cause (from子句)
            elif cause is None:
                cause = convert_expression(child)
        # 检查是否有from关键字，确保下一个表达式是cause
        elif child.type == "from" and i + 1 < len(node.children):
            next_child = node.children[i + 1]
            if next_child.type in ["identifier", "call", "expression"]:
                cause = convert_expression(next_child)
    
    return RaiseStatement(NodeType.RAISE_STMT, expression, cause)

def convert_with_statement(node: tree_sitter.Node) -> WithStatement:
    """
    将Python的with语句转换为内部表示的WithStatement
    
    支持以下形式:
    1. 单个资源: with open('file.txt') as f:
    2. 多个资源: with open('file1.txt') as f1, open('file2.txt') as f2:
    3. 不带as的资源: with lock:
    """
    # 获取with_clause节点
    with_clause = None
    body_node = None
    is_async = False
    
    # 遍历子节点查找with_clause和body
    for child in node.children:
        if child.type == "with_clause":
            with_clause = child
        elif child.type == "block":
            body_node = child
        elif child.type == "async":
            is_async = True
    
    # 处理body
    body = convert_statement(body_node) if body_node else None
    
    # 处理资源
    resources = []
    if with_clause:
        # 遍历with_clause的子节点，查找with_item
        for child in with_clause.children:
            if child.type == "with_item":
                # 获取资源表达式
                value_node = child.child_by_field_name("value")
                alias_node = child.child_by_field_name("alias")
                
                # 如果没有通过field_name找到，尝试通过子节点类型查找
                if not value_node:
                    for item_child in child.children:
                        if item_child.type in ["call", "identifier", "attribute", "expression"]:
                            value_node = item_child
                            break
                
                # 处理as_pattern
                if not alias_node and not value_node:
                    for item_child in child.children:
                        if item_child.type == "as_pattern":
                            # 从as_pattern中获取value和alias
                            as_children = item_child.children
                            if len(as_children) >= 3:  # 至少需要value, as, alias
                                value_node = as_children[0]
                                alias_node = as_children[2]
                
                # 转换资源表达式
                if value_node:
                    resource = convert_expression(value_node)
                    
                    # 如果有别名，创建带as关系的表达式
                    if alias_node:
                        alias = convert_expression(alias_node)
                        if alias and resource:
                            # 使用自定义的FieldAccess来表示"as"关系
                            # 直接创建FieldAccess对象，不使用node_factory
                            resource = FieldAccess(
                                NodeType.FIELD_ACCESS,
                                resource,
                                alias,
                                FieldAccessOps.AS,
                                False,
                                True  # is_as_pattern=True
                            )
                    
                    if resource:
                        resources.append(resource)
    
    # 如果有多个资源，创建数组表达式
    if len(resources) > 1:
        expr_list = ExpressionList(NodeType.EXPRESSION_LIST, resources)
        resource_array = ArrayExpression(NodeType.ARRAY_EXPR, expr_list)
        with_stmt = WithStatement(NodeType.WITH_STMT, resource_array, body)
    elif len(resources) == 1:
        # 单个资源
        with_stmt = WithStatement(NodeType.WITH_STMT, resources[0], body)
    else:
        # 没有资源，创建一个空的标识符作为占位符
        empty_resource = Identifier(NodeType.IDENTIFIER, "_empty_resource")
        with_stmt = WithStatement(NodeType.WITH_STMT, empty_resource, body)
    
    # 设置异步标志
    with_stmt.is_async = is_async
    return with_stmt
def convert_try_statement(node: tree_sitter.Node) -> TryStatement:
    """
    将Python try语句转换为内部表示的TryStatement
    """
    body_node = node.child_by_field_name("body")
    body = convert_statement(body_node) if body_node else None
    catch_clauses = []
    finally_block = None
    else_block = None
    
    for child in node.children:
        if child.type == "except_clause":
            # 获取异常类型和别名
            exception_type = None
            exception_var = None
            except_body = None
            
            # 处理异常类型
            for except_child in child.children:
                if except_child.type == "type" or except_child.type == "expression":
                    # 直接处理异常类型表达式
                    exception_type = convert_expression(except_child)
                elif except_child.type == "identifier" and exception_type is None:
                    # 处理简单标识符作为异常类型
                    exception_type = convert_expression(except_child)
                elif except_child.type == "as_pattern":
                    # 处理 as 模式 (例如 except Exception as e:)
                    as_type = None
                    as_name = None
                    
                    for as_child in except_child.children:
                        if as_child.type == "expression" and as_type is None:
                            as_type = convert_expression(as_child)
                        elif as_child.type == "identifier" and as_type is None:
                            as_type = convert_expression(as_child)
                        elif as_child.type == "as_pattern_target":
                            as_name = convert_expression(as_child.child(0))
                    
                    # 设置异常类型和变量
                    if as_type:
                        exception_type = as_type
                    if as_name:
                        exception_var = as_name
                elif except_child.type == "block":
                    except_body = convert_statement(except_child)
            
            # 创建CatchClause
            catch_clause = node_factory.create_catch_clause(
                except_body,
                exception_type,  # 直接使用exception_type
                exception_var,
                None
            )
            catch_clauses.append(catch_clause)
        
        elif child.type == "finally_clause":
            for finally_child in child.children:
                if finally_child.type == "block":
                    finally_block = convert_statement(finally_child)
                    break
        
        elif child.type == "else_clause":
            for else_child in child.children:
                if else_child.type == "block":
                    else_block = convert_statement(else_child)
                    break
    
    # 创建TryHandlers对象，包含catch_clauses
    handlers = node_factory.create_try_handlers(catch_clauses)
    
    # 如果有finally块，将其添加到handlers中
    if finally_block:
        handlers.finally_block = finally_block
    
    # 创建并返回TryStatement，包含body、handlers和else_block
    return node_factory.create_try_stmt(body, handlers, else_block)

def convert_identifier(node: tree_sitter.Node) -> Identifier:
    """
    将Python标识符转换为内部表示的Identifier
    """
    return Identifier(NodeType.IDENTIFIER, node.text.decode('utf-8'))

def convert_literal(node: tree_sitter.Node) -> Literal:
    """
    将Python字面量转换为内部表示的Literal
    """
    if node.type == "string":
        # 处理字符串字面量
        content = ""
        # 获取原始字符串文本，包括引号
        raw_text = node.text.decode('utf-8')
        
        # 检查字符串的引号类型
        quote_type = None
        if raw_text.startswith("'") and raw_text.endswith("'"):
            quote_type = "single"
        elif raw_text.startswith('"') and raw_text.endswith('"'):
            quote_type = "double"
        elif raw_text.startswith("'''") and raw_text.endswith("'''"):
            quote_type = "triple_single"
        elif raw_text.startswith('"""') and raw_text.endswith('"""'):
            quote_type = "triple_double"
        
        # 获取字符串内容
        for child in node.children:
            if child.type == "string_content":
                content = child.text.decode('utf-8')
                break
        
        # 创建带有引号类型信息的字面量
        literal = node_factory.create_literal(content)
        literal.is_string = True
        literal.quote_type = quote_type
        literal.raw_text = raw_text  # 保存原始文本，包括引号
        return literal
    
    elif node.type in ["integer", "float"]:
        # 处理数字字面量
        value = node.text.decode()
        literal = node_factory.create_literal(value)
        literal.is_string = False
        return literal
    
    elif node.type == "true":
        literal = node_factory.create_literal("True")
        literal.is_string = False
        return literal
    
    elif node.type == "false":
        literal = node_factory.create_literal("False")
        literal.is_string = False
        return literal
    
    elif node.type == "none":
        literal = node_factory.create_literal("None")
        literal.is_string = False
        return literal
    
    else:
        # 其他类型的字面量
        value = node.text.decode('utf-8')
        # 检查是否是字符串形式的数字
        is_string = False
        if value.startswith("'") and value.endswith("'"):
            is_string = True
        elif value.startswith('"') and value.endswith('"'):
            is_string = True
        
        literal = node_factory.create_literal(value)
        literal.is_string = is_string
        return literal

def convert_binary_operator(node: tree_sitter.Node) -> BinaryExpression:
    """
    将Python二元运算符转换为内部表示的BinaryExpression
    """
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")
    operator_node = node.child_by_field_name("operator")
    
    left = convert_expression(left_node) if left_node else None
    right = convert_expression(right_node) if right_node else None
    operator = operator_node.text.decode('utf-8') if operator_node else "+"
    op = get_binary_op(operator)
    return BinaryExpression(NodeType.BINARY_EXPR, left, right, op)


def convert_boolean_operator(node: tree_sitter.Node) -> BinaryExpression:
    """
    将Python布尔运算符转换为内部表示的BooleanExpression
    """
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")
    operator_node = node.child_by_field_name("operator")
    
    left = convert_expression(left_node) if left_node else None
    right = convert_expression(right_node) if right_node else None
    
    operator_text = operator_node.text.decode('utf-8') if operator_node else "and"

    op = get_binary_op(operator_text)
    return BinaryExpression(NodeType.BINARY_EXPR, left, right, op)


def convert_call(node: tree_sitter.Node) -> CallExpression:
    """
    将Python函数调用转换为内部表示的CallExpression
    """
    function_node = node.child_by_field_name("function")
    arguments_node = node.child_by_field_name("arguments")
    
    function = convert_expression(function_node) if function_node else None
    
    arguments = []
    if arguments_node:
        for child in arguments_node.children:

            # 跳过括号和逗号
            if child.type in [",", "(", ")"]:
                continue
                
            if child.type != "expression_list":
                arg = convert_expression(child)
                if arg is not None:
                    arguments.append(arg)
            # 处理可能嵌套在expression中的参数
            elif child.type == "expression_list":
                for expr_child in child.children:
                    if expr_child.type != ",":
                        expr = convert_expression(expr_child)
                        if expr is not None:
                            arguments.append(expr)
    # 创建ExpressionList并添加NodeType.CALL_EXPR
    args_list = ExpressionList(NodeType.EXPRESSION_LIST, arguments)
    return CallExpression(NodeType.CALL_EXPR, function, args_list, False)

def convert_attribute(node: tree_sitter.Node) -> FieldAccess:
    """
    将Python属性访问转换为内部表示的FieldAccess
    处理嵌套的方法调用链，如obj.method1().method2()
    """
    object_node = node.child_by_field_name("object")
    attribute_node = node.child_by_field_name("attribute")
    
    # 递归处理对象，可能是另一个属性访问或函数调用
    object_expr = convert_expression(object_node) if object_node else None
    
    # 处理属性名
    attribute = None
    if attribute_node:
        if attribute_node.type == "identifier":
            attribute = Identifier(NodeType.IDENTIFIER, attribute_node.text.decode('utf-8'))
        else:
            attribute = convert_expression(attribute_node)
    
    # 如果对象或属性为空，提供合理的默认值
    if object_expr is None:
        #logger.warning(f"属性访问中缺少对象: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
        object_expr = Identifier(NodeType.IDENTIFIER, "_obj")
    
    if attribute is None:
        #logger.warning(f"属性访问中缺少属性名: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
        attribute = Identifier(NodeType.IDENTIFIER, "_attr")
    
    return FieldAccess(NodeType.FIELD_ACCESS, object_expr, attribute)

#NOTE:需要再检查
def convert_subscript(node: tree_sitter.Node) -> ArrayAccess:
    """
    将Python下标访问转换为内部表示的ArrayAccess
    处理复杂的下标访问，包括多维切片和省略号
    """
    value_node = node.child_by_field_name("value")
    subscript_nodes = []
    
    # 收集所有的下标节点
    for child in node.children:
        if child.type not in ["[", "]", ","]:
            if child != value_node:
                subscript_nodes.append(child)
    
    # 递归处理被访问的对象
    value = convert_expression(value_node) if value_node else None
    
    # 处理下标表达式
    elements = []
    for subscript_node in subscript_nodes:
        if subscript_node.type == "slice":
            # 处理切片操作
            slice_elements = []
            start_node = subscript_node.child_by_field_name("start")
            end_node = subscript_node.child_by_field_name("end")
            step_node = subscript_node.child_by_field_name("step")
            
            # 处理start, end, step
            slice_elements.append(convert_expression(start_node) if start_node else Literal(NodeType.LITERAL, "None"))
            slice_elements.append(convert_expression(end_node) if end_node else Literal(NodeType.LITERAL, "None"))
            if step_node or any(child.type == ":" for child in subscript_node.children[1:]):
                slice_elements.append(convert_expression(step_node) if step_node else Literal(NodeType.LITERAL, "None"))
            
            expr_list = ExpressionList(NodeType.EXPRESSION_LIST, slice_elements)
            elements.append(ArrayExpression(NodeType.ARRAY_EXPR, expr_list))
            
        elif subscript_node.type == "ellipsis":
            # 处理省略号
            elements.append(Literal(NodeType.LITERAL, "Ellipsis"))
            
        else:
            # 处理普通索引
            expr = convert_expression(subscript_node)
            if expr:
                elements.append(expr)
    
    # 创建最终的下标表达式
    if len(elements) == 1:
        subscript = elements[0]
    else:
        expr_list = ExpressionList(NodeType.EXPRESSION_LIST, elements)
        subscript = ArrayExpression(NodeType.ARRAY_EXPR, expr_list)
    
    # 处理默认值
    if value is None:
        value = Identifier(NodeType.IDENTIFIER, "_obj")
    
    return ArrayAccess(NodeType.ARRAY_ACCESS, value, subscript)



def convert_list(node: tree_sitter.Node) -> ArrayExpression:
    """
    将Python列表转换为内部表示的ArrayExpression
    """
    elements = []
    for child in node.children:
        if child.type not in ["[", "]", ","]:
            element = convert_expression(child)
            if element is not None:
                elements.append(element)
    
    # 修复：添加NodeType.ARRAY_EXPR作为第一个参数
    expr_list = ExpressionList(NodeType.EXPRESSION_LIST, elements)
    return ArrayExpression(NodeType.ARRAY_EXPR, expr_list, ArrayPatternType.LIST)


def convert_list_comprehension(node: tree_sitter.Node) -> ComprehensionExpression:
    """
    将Python列表推导式转换为内部表示的ComprehensionExpression
    """
    # 获取主体表达式（通常是第一个子节点）
    body_node = None
    clauses = []
    
    # 遍历子节点，找出主体表达式和各个子句
    for child in node.children:
        if child.type == "[" or child.type == "]":
            continue
        elif child.type == "for_in_clause":
            # 处理for子句
            left_node = child.child_by_field_name("left")
            right_node = child.child_by_field_name("right")
            
            left = convert_expression(left_node) if left_node else None
            right = convert_expression(right_node) if right_node else None
            
            clause = ComprehensionClause(
                NodeType.COMPREHENSION_CLAUSE, 
                is_for=True,
                left=left,
                right=right
            )
            clauses.append(clause)
        elif child.type == "if_clause":
            # 处理if子句
            condition_node = child.child(1) if child.child_count > 1 else None
            condition = convert_expression(condition_node) if condition_node else None
            
            clause = ComprehensionClause(
                NodeType.COMPREHENSION_CLAUSE,
                is_for=False,
                condition=condition
            )
            clauses.append(clause)
        elif body_node is None:
            # 第一个非特殊节点被视为主体表达式
            body_node = child
    
    # 转换主体表达式
    body = convert_expression(body_node) if body_node else None
    
    if body is None:
        #logger.warning(f"列表推导式缺少主体表达式: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
        return None
    
    # 创建ComprehensionExpression
    return ComprehensionExpression(
        NodeType.COMPREHENSION_EXPR,
        body,
        clauses,
        comprehension_type="list"
    )



def convert_tuple(node: tree_sitter.Node) -> ArrayExpression:
    """
    将Python元组转换为内部表示的ArrayExpression
    """
    elements = []
    for child in node.children:
        if child.type in ["expression", "list_splat"]:
            element = convert_expression(child)
            if element is not None:
                elements.append(element)
    
    # 修复：添加NodeType.ARRAY_EXPR作为第一个参数
    expr_list = ExpressionList(NodeType.EXPRESSION_LIST, elements)
    return ArrayExpression(NodeType.ARRAY_EXPR, expr_list, ArrayPatternType.TUPLE)

def convert_dictionary(node: tree_sitter.Node) -> Expression:
    """
    将Python字典转换为内部表示的Expression
    """
    # 这里可以使用ObjectExpression或者其他适合表示字典的节点类型
    # 暂时使用ArrayExpression表示
    elements = []
    for child in node.children:
        if child.type == "pair":
            key_node = child.child_by_field_name("key")
            value_node = child.child_by_field_name("value")
            
            key = convert_expression(key_node) if key_node else None
            value = convert_expression(value_node) if value_node else None
            
            if key is not None and value is not None:
                # 将键值对表示为二元表达式
                elements.append(BinaryExpression(NodeType.BINARY_EXPR, key, value, BinaryOps.KEY_VALUE))
    
    return ArrayExpression(NodeType.ARRAY_EXPR, ExpressionList(NodeType.EXPRESSION_LIST, elements), ArrayPatternType.DICT)


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
    return ArrayExpression(NodeType.ARRAY_EXPR, ExpressionList(NodeType.EXPRESSION_LIST, elements), ArrayPatternType.SET)

'''
def convert_parenthesized_expression(node: tree_sitter.Node) -> ParenthesizedExpression:
    """
    将Python括号表达式转换为内部表示的ParenthesizedExpression
    """
    for child in node.children:
        if child.type != "(" and child.type != ")":
            expr = convert_expression(child)
            if expr is not None:
                return ParenthesizedExpression(expr)
    
    return None
'''
def convert_parenthesized_expression(node: tree_sitter.Node) -> ParenthesizedExpression:
    if node.child_count != 3:
        raise AssertionError("parenthesized_expression should have 3 children")

    expr = convert_expression(node.children[1])
    return node_factory.create_parenthesized_expr(expr)

def convert_assignment(node: tree_sitter.Node) -> Expression:
    """
    将Python赋值表达式转换为内部表示的AssignmentExpression或其他表达式
    """
    
    if node.child_count == 1:
        # 单个子节点，可能是函数调用或其他表达式
        expr_node = node.children[0]
        expr = convert_expression(expr_node)
        # 直接返回转换后的表达式
        return expr
    elif node.child_count == 5:
        #类型注解
        pass
    
    else:
        # 赋值表达式
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")
        
        left = convert_expression(left_node) if left_node else None
        right = convert_expression(right_node) if right_node else None
        
        return AssignmentExpression(NodeType.ASSIGNMENT_EXPR, left, right, get_assignment_op("="))

    
    # 处理其他情况，返回一个默认值
    #logger.warning(f"无法处理的赋值表达式: {node.type} 子节点数: {node.child_count} 文本内容：{node.text.decode()}")
    return node_factory.create_literal("None")


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
    op = get_assignment_op(operator)
    return AssignmentExpression(NodeType.ASSIGNMENT_EXPR, left, right, op)

def convert_unary_operator(node: tree_sitter.Node) -> Expression:
    """
    将Python一元运算符转换为内部表示的UnaryExpression
    """
    argument_node = node.child_by_field_name("argument")
    operator_node = node.child_by_field_name("operator")
    
    argument = convert_expression(argument_node) if argument_node else None
    operator = operator_node.text.decode('utf-8') if operator_node else "-"
    
    # 检查是否是负数字面量
    if argument and argument.node_type == NodeType.LITERAL and operator == "-":
        # 如果是数字字面量，直接创建一个新的带负号的字面量
        try:
            # 尝试将值解析为数字
            value = argument.value
            if value.isdigit() or (value.replace('.', '', 1).isdigit() and value.count('.') <= 1):
                return node_factory.create_literal(f"-{value}")
        except (AttributeError, ValueError):
            pass
    
    # 其他情况正常处理
    op = get_unary_op(operator)
    return node_factory.create_unary_expr(argument, op)

def convert_comparison_operator(node: tree_sitter.Node) -> BinaryExpression:
    """
    将Python比较运算符转换为内部表示的BinaryExpression
    """
    left_node = node.children[0]
    right_node = node.children[-1]

    # 获取运算符节点
    operator_nodes = []
    for i in range(1, len(node.children) - 1):
        if node.children[i].type not in [",", "(", ")"]:
            operator_nodes.append(node.children[i])
    
    left = convert_expression(left_node) if left_node else None
    right = convert_expression(right_node) if right_node else None
    
    # 获取操作符，处理可能有多个运算符节点的情况
    operator = "=="  # 默认值
    if operator_nodes:
        # 处理复合运算符
        if len(operator_nodes) > 1:
            # 处理"is not"运算符
            if operator_nodes[0].type == "is not":
                operator = "isnot"
            # 处理"not in"运算符
            elif operator_nodes[0].type == "not in":
                operator = "notin"

        else:
            # 单个运算符节点
            operator = operator_nodes[0].text.decode('utf-8')
    
    # 处理Python特有的操作符
    if operator in ["in", "not in"]:
        # 对于in和not in操作符，确保左右操作数存在
        if left is None:
            #logger.warning(f"比较表达式中缺少左操作数，使用默认值替代: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
            # 对于in操作符，左侧通常是一个值，使用一个空字符串作为默认值
            left = Literal(NodeType.LITERAL, "")  # 修正参数顺序
        
        if right is None:
            #logger.warning(f"比较表达式中缺少右操作数，使用默认值替代: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
            # 对于in操作符，右侧通常是一个容器，使用一个空列表作为默认值
            expr_list = ExpressionList(NodeType.EXPRESSION_LIST, [])
            right = ArrayExpression(NodeType.ARRAY_EXPR, expr_list)
    
    # 处理is和is not操作符
    elif operator in ["is", "is not"]:
        # 处理特殊情况：is None 或 is not None
        if right is None or (isinstance(right, Identifier) and right.name == "None"):
            # 创建一个表示None的字面量
            right = Literal(NodeType.LITERAL, "None")  # 修正参数顺序
        
        if left is None:
            #logger.warning(f"比较表达式中缺少左操作数，使用默认值替代: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
            # 对于is操作符，左侧通常是一个变量，使用一个默认标识符
            left = Identifier(NodeType.IDENTIFIER, "_")
    
    # 处理其他比较操作符
    else:
        # 确保左右操作数不为None，使用更合理的默认值
        if left is None:
            #logger.warning(f"比较表达式中缺少左操作数，使用默认值替代: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
            # 使用True作为默认值，而不是"undefined"
            left = Literal(NodeType.LITERAL, "True")  # 修正参数顺序和类型
        
        if right is None:
            #logger.warning(f"比较表达式中缺少右操作数，使用默认值替代: {node.text.decode('utf-8') if hasattr(node, 'text') else 'unknown'}")
            # 使用True作为默认值，而不是"undefined"
            right = Literal(NodeType.LITERAL, "True")  # 修正参数顺序和类型
    
    op = get_binary_op(operator)
  
    # 使用node_factory创建BinaryExpression，确保接口一致
    return node_factory.create_binary_expr(left, right, op)

def convert_expression_list(node: tree_sitter.Node) -> ExpressionList:
    """
    将Python表达式列表转换为内部表示的ExpressionList
    """
    expressions = []
    for child in node.children:
        if child.type == "expression":
            expr = convert_expression(child)
            if expr is not None:
                expressions.append(expr)
    
    return ExpressionList(NodeType.EXPRESSION_LIST, expressions)

def convert_keyword_argument(node: tree_sitter.Node) -> Expression:
    """
    处理关键字参数，例如 func(key=value)
    """
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")
    
    if not name_node or not value_node:
        return node_factory.create_literal("None")
    
    name = name_node.text.decode('utf-8')
    value = convert_expression(value_node)
    
    # 创建关键字参数节点
    name_id = node_factory.create_identifier(name)
    # 使用现有的方法创建一个表示关键字参数的节点
    # 可以使用BinaryExpression来表示key=value关系
    return BinaryExpression(NodeType.BINARY_EXPR, name_id, value, BinaryOps.EQD)

def convert_dictionary_splat(node: tree_sitter.Node) -> Expression:
    """
    处理字典展开操作，例如 **kwargs
    """
    argument_node = node.children[-1]
    if not argument_node:
        return node_factory.create_literal("None")
    
    argument = convert_expression(argument_node)
    
    # 创建字典展开元素
    return node_factory.create_spread_element(argument, True)

def convert_elif_clause(node: tree_sitter.Node) -> Statement:
    """
    处理elif子句，将其转换为if语句的alternate部分
    """
    condition_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    alternative_node = node.child_by_field_name("alternative")
    
    if not condition_node or not consequence_node:
        return node_factory.create_empty_stmt()
    
    condition = convert_expression(condition_node)
    consequence = convert_statement(consequence_node)
    
    # 处理可能的下一个elif或else子句
    alternative = None
    if alternative_node:
        alternative = convert_statement(alternative_node)

    # 创建if语句作为elif的表示
    return node_factory.create_if_stmt(condition, consequence, alternative)

def convert_not_operator(node: tree_sitter.Node) -> Expression:
    """
    处理not运算符，例如 not condition
    """
    argument_node = node.child_by_field_name("argument")
    if not argument_node:
        return node_factory.create_literal("None")
    
    argument = convert_expression(argument_node)
    
    # 创建一元表达式，使用NOT作为操作符
    return node_factory.create_unary_expr(argument, UnaryOps.PYNOT)

def convert_else_clause(node: tree_sitter.Node) -> Statement:
    """
    处理else子句，将其转换为内部表示的Statement
    """
    body_node = node.child_by_field_name("body")
    if not body_node:
        return node_factory.create_empty_stmt()
    
    # 直接返回else子句的主体
    return convert_statement(body_node)

def convert_list_splat(node: tree_sitter.Node) -> Expression:
    """
    处理列表展开和字典展开操作，例如 *args **args
    """

    argument_node = node.children[-1]

    
    argument = convert_expression(argument_node)
    
    # 创建列表展开元素
    return node_factory.create_spread_element(argument, False)

# 在文件末尾添加条件表达式转换函数
def convert_conditional_expression(node: tree_sitter.Node) -> Expression:
    """
    处理Python条件表达式（三元运算符），例如 value_if_true if condition else value_if_false
    """
    # 根据tree-sitter的Python语法树结构，条件表达式有三个主要部分
    # 但它们的顺序是：consequence if condition else alternative
    
    # 查找"if"和"else"关键字的位置
    if_index = -1
    else_index = -1
    
    for i, child in enumerate(node.children):
        if child.type == "if":
            if_index = i
        elif child.type == "else":
            else_index = i
    
    # 如果找不到if或else关键字，则无法处理
    if if_index == -1 or else_index == -1 or if_index >= else_index:
        record_unhandled_type(node.type, "条件表达式", getattr(record_unhandled_type, 'instance_id', None))
        return node_factory.create_literal("None")
    
    # 收集if前面的所有节点作为consequence
    consequence_nodes = []
    for i in range(0, if_index):
        consequence_nodes.append(node.children[i])
    
    # 收集if和else之间的所有节点作为condition
    condition_nodes = []
    for i in range(if_index + 1, else_index):
        condition_nodes.append(node.children[i])
    
    # 收集else后面的所有节点作为alternative
    alternative_nodes = []
    for i in range(else_index + 1, len(node.children)):
        alternative_nodes.append(node.children[i])
    
    # 处理consequence部分
    consequence = None
    if consequence_nodes:
        # 如果只有一个节点，直接转换
        if len(consequence_nodes) == 1:
            consequence = convert_expression(consequence_nodes[0])
        else:
            stmts = []
            for node in consequence_nodes:
                expr = convert_expression(node)
                stmts.append(expr)
                stmts = node_factory.create_statement_list(stmts)
                if expr is not None:
                    consequence = expr
                    break
    
    # 处理condition部分
    condition = None
    if condition_nodes:
        if len(condition_nodes) == 1:
            condition = convert_expression(condition_nodes[0])
        else:
            for node in condition_nodes:
                expr = convert_expression(node)
                if expr is not None:
                    condition = expr
                    break
    
    # 处理alternative部分
    alternative = None
    if alternative_nodes:
        if len(alternative_nodes) == 1:
            alternative = convert_expression(alternative_nodes[0])
        else:
            for node in alternative_nodes:
                expr = convert_expression(node)
                if expr is not None:
                    alternative = expr
                    break
    
    # 如果任何必要部分缺失，返回一个默认值
    if condition is None or consequence is None or alternative is None:
        record_unhandled_type(node.type, "条件表达式缺少部分", getattr(record_unhandled_type, 'instance_id', None))
        return node_factory.create_literal("None")
    
    lhs = consequence
    rhs = BinaryExpression(
        NodeType.BINARY_EXPR,
        condition,
        alternative,
        get_binary_op("else")
    )
    op = get_binary_op("if")
    return node_factory.create_binary_expr(lhs, rhs, op)

def convert_block_statement(node: tree_sitter.Node) -> BlockStatement:
    stmts = []
    for stmt_node in node.children:
        stmts.append(convert_statement(stmt_node))
    stmts = node_factory.create_statement_list(stmts)
    return node_factory.create_block_stmt(stmts)

def convert_pattern_list(node: tree_sitter.Node) -> Expression:
    """
    处理Python中的模式列表，用于多重赋值，例如 a, b, c = some_function()
    使用ArrayExpression来表示多重赋值的目标变量列表
    """
    identifiers = []
    
    # 遍历所有子节点，收集标识符
    for child in node.children:
        # 跳过逗号
        if child.type == ",":
            continue
        
        # 转换标识符或其他模式
        expr = convert_expression(child)
        if expr is not None:
            identifiers.append(expr)
    
    # 创建一个ExpressionList
    expr_list = ExpressionList(NodeType.EXPRESSION_LIST, identifiers)
    
    return ArrayExpression(NodeType.ARRAY_EXPR, expr_list, ArrayPatternType.PATTERN)


def convert_lambda(node: tree_sitter.Node) -> LambdaExpression:
    """
    将Python lambda表达式转换为内部表示的LambdaExpression
    处理各种形式的lambda表达式，包括：
    - 简单lambda: lambda x: x ** 2
    - 条件表达式lambda: lambda x: "even" if x % 2 == 0 else "odd"
    - 嵌套lambda: lambda x: lambda y: x + y
    - 多参数lambda: lambda a, b: a + b
    """
    # 处理参数
    params_node = node.child_by_field_name("parameters")
    body_node = node.child_by_field_name("body")
    
    # 处理参数列表
    params = []
    if params_node:
        for child in params_node.children:
            if child.type == ",":
                continue
            if child.type == "identifier":
                param = convert_identifier(child)
                param_decl = node_factory.create_variable_declarator(param)
                params.append(node_factory.create_untyped_param(param_decl))
    
    # 创建参数列表
    param_list = node_factory.create_formal_parameter_list(params)
    
    # 处理lambda函数体
    body = None
    if body_node:
        # 处理不同类型的lambda函数体
        if body_node.type == "lambda":
            # 嵌套lambda
            body = convert_lambda(body_node)
        elif body_node.type == "conditional_expression":
            # 条件表达式
            body = convert_conditional_expression(body_node)
        else:
            # 普通表达式
            body = convert_expression(body_node)
    
    # 创建并返回LambdaExpression
    return node_factory.create_lambda_expr(param_list, body, False)

def convert_as_pattern(node: tree_sitter.Node) -> Expression:
    """
    将Python的as模式转换为内部表示
    
    处理形如 "expression as identifier" 的模式，常见于with语句和except子句中
    """
    value_node = None
    alias_node = None
    
    # 遍历子节点查找value和alias
    for i, child in enumerate(node.children):
        if i == 0:  # 第一个子节点通常是value
            value_node = child
        elif child.type == "as":
            continue  # 跳过as关键字
        elif i > 1 and alias_node is None:  # as关键字后的节点是alias
            alias_node = child
    
    # 转换value和alias
    value = convert_expression(value_node) if value_node else None
    alias = convert_expression(alias_node) if alias_node else None
    
    # 如果value和alias都存在，创建FieldAccess表示as关系
    if value and alias:
        return FieldAccess(
            NodeType.FIELD_ACCESS,
            value,
            alias,
            FieldAccessOps.AS,
            False,
            True  # is_as_pattern=True
        )
    
    # 如果缺少value或alias，返回可用的部分
    return value if value else (alias if alias else None)

def convert_as_pattern_target(node: tree_sitter.Node) -> Expression:
    """
    将Python的as_pattern_target转换为内部表示
    
    as_pattern_target通常是as关键字后面的标识符
    """
    # 通常as_pattern_target只有一个子节点，即标识符
    if node.child_count > 0:
        return convert_expression(node.child(0))
    
    # 如果没有子节点，尝试直接转换节点本身
    if node.type == "identifier":
        return Identifier(NodeType.IDENTIFIER, node.text.decode('utf-8'))
    
    # 无法处理的情况
    return None

'''
def convert_concatenated_string(node: tree_sitter.Node) -> Expression:
    """
    将Python的concatenated_string节点转换为内部表示
    
    处理形如 'string1' 'string2' 的字符串拼接
    """
    # 创建一个空的字符串字面量作为初始值
    result = None
    
    # 遍历所有子节点（字符串部分）
    for child in node.children:
        # 转换每个字符串部分
        if child.type == "string":
            string_expr = convert_expression(child)
            
            if result is None:
                # 第一个字符串部分
                result = string_expr
            else:
                # 使用二元表达式表示字符串拼接
                result = BinaryExpression(
                    NodeType.BINARY_EXPR,
                    result,
                    string_expr,
                    BinaryOps.PLUS  # 使用加号操作符表示字符串拼接
                )
    
    # 如果没有找到任何字符串部分，返回一个空字符串字面量
    if result is None:
        result = Literal(NodeType.LITERAL, "")
    
    return result
'''

def convert_concatenated_string(node: tree_sitter.Node) -> Expression:
    """
    将Python的concatenated_string节点转换为内部表示
    
    处理形如 'string1' 'string2' 的字符串拼接
    """
    # 收集所有字符串部分
    string_parts = []
    
    # 遍历所有子节点（字符串部分）
    for child in node.children:
        # 转换每个字符串部分
        if child.type == "string":
            string_expr = convert_expression(child)
            if string_expr:
                string_parts.append(string_expr)
    
    # 如果没有找到任何字符串部分，返回一个空字符串字面量
    if not string_parts:
        return Literal(NodeType.LITERAL, "")
    
    # 创建一个特殊的数组表达式来表示连接的字符串
    # 使用is_concatenated_string标记这是一个字符串连接
    expr_list = ExpressionList(NodeType.EXPRESSION_LIST, string_parts)
    array_expr = ArrayExpression(NodeType.ARRAY_EXPR, expr_list, ArrayPatternType.CONCATENATED_STRING)
    
    # 设置一个特殊属性，表示这是一个连接的字符串
    array_expr.is_concatenated_string = True
    
    return array_expr