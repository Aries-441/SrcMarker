from .common import BaseStringifier
from mutable_tree.nodes import NodeType
from mutable_tree.nodes.statements.declarators import Declarator
from mutable_tree.nodes.statements.func_declaration import FunctionHeader, FunctionDeclaration, FormalParameterList
from mutable_tree.nodes import Statement, Expression, BlockStatement, StatementList, ExpressionList
from mutable_tree.nodes import Identifier, Literal
from mutable_tree.nodes.expressions.binary_expr import BinaryExpression
from mutable_tree.nodes.expressions.unary_expr import UnaryExpression
from mutable_tree.nodes.expressions.call_expr import CallExpression
from mutable_tree.nodes.expressions.field_access import FieldAccess
from mutable_tree.nodes.expressions.array_expr import ArrayExpression, ArrayPatternType
from mutable_tree.nodes.expressions.array_access import ArrayAccess
from mutable_tree.nodes.statements.if_stmt import IfStatement
from mutable_tree.nodes.statements.for_in_stmt import ForInStatement, ForInType
from mutable_tree.nodes.statements.while_stmt import WhileStatement
from mutable_tree.nodes.statements.try_stmt import TryStatement, CatchClause, FinallyClause
from mutable_tree.nodes.statements.return_stmt import ReturnStatement
from mutable_tree.nodes.statements.expression_stmt import ExpressionStatement
from mutable_tree.nodes.statements.pass_stmt import PassStatement
from mutable_tree.nodes.statements.assert_stmt import AssertStatement
from mutable_tree.nodes.statements.raise_stmt import RaiseStatement
from mutable_tree.nodes.statements.with_statement import WithStatement

import logging
logger = logging.getLogger(__name__)


class PythonStringifier(BaseStringifier):
    def __init__(self):
        super().__init__()
    
    def _indent(self, text: str, indent_size: int = 4) -> str:
        """
        对文本进行缩进处理
        """
        if not text:
            return ""
        
        indent = " " * indent_size
        lines = text.split("\n")
        indented_lines = [indent + line for line in lines]
        return "\n".join(indented_lines)
    
    def stringify_Declarator(self, node: Declarator) -> str:
        """将Declarator对象转换为字符串"""
        if hasattr(node, 'decl_id') and node.decl_id is not None:
            return self.stringify(node.decl_id)
        elif hasattr(node, 'declarator') and node.declarator is not None:
            return self.stringify(node.declarator)
        return ""
    
    def stringify_VariableDeclarator(self, node) -> str:
        """处理变量声明器"""
        if hasattr(node, 'decl_id') and node.decl_id is not None:
            return self.stringify(node.decl_id)
        return ""
    
    def stringify_InitializingDeclarator(self, node) -> str:
        """处理带初始化的声明器"""
        name = self.stringify(node.declarator) if hasattr(node, 'declarator') and node.declarator else ""
        initializer = ""
        
        if hasattr(node, 'value') and node.value:
            initializer = f"={self.stringify(node.value)}"
        
        return f"{name}{initializer}"
    
    def stringify_FormalParameterList(self, node: FormalParameterList) -> str:
        """处理形参列表"""
        params = []
        for param in node.get_children():
            param_str = self.stringify(param)
            if param_str:
                params.append(param_str)
        
        return ", ".join(params)
    
    def stringify_UntypedParameter(self, node) -> str:
        """处理无类型参数"""
        if hasattr(node, 'declarator') and node.declarator is not None:
            return self.stringify(node.declarator)
        return ""
    
    def stringify_SpreadParameter(self, node) -> str:
        """处理展开参数，如 *args 或 **kwargs"""
        if not hasattr(node, 'declarator') or not node.declarator:
            return "*args"  # 默认值
        
        param_name = self.stringify(node.declarator)
        
        # 判断是否为字典展开参数
        # 这里需要根据实际情况调整判断逻辑
        is_dict_spread = False
        if hasattr(node, 'is_dict_spread'):
            is_dict_spread = node.is_dict_spread
        
        prefix = "**" if is_dict_spread else "*"
        return f"{prefix}{param_name}"
    
    def stringify_FunctionDeclarator(self, node) -> str:
        """处理函数声明器"""
        name = ""
        if hasattr(node, 'declarator') and node.declarator is not None:
            name = self.stringify(node.declarator)
        
        params = ""
        if hasattr(node, 'parameters') and node.parameters is not None:
            params = self.stringify(node.parameters)
        
        return f"def {name}({params})"
    
    def stringify_FunctionHeader(self, node: FunctionHeader) -> str:
        """处理函数头"""
        if hasattr(node, 'func_decl') and node.func_decl is not None:
            return self.stringify(node.func_decl)
        return ""
    
    def stringify_FunctionDeclaration(self, node: FunctionDeclaration) -> str:
        """
        将FunctionDeclaration对象转换为字符串
        """
        # 处理函数头
        header = self.stringify(node.header) if node.header else ""
        
        # 处理函数体
        body = self.stringify(node.body) if node.body else "pass"
        
        # 如果函数体不是以换行开始，添加换行和缩进
        if body and not body.startswith("\n"):
            body = "\n" + self._indent(body)
        
        # 注意：不要在末尾添加分号
        return f"{header}:{body}"
    

    
    def stringify_BlockStatement(self, node: BlockStatement) -> str:
        """处理块语句"""
        stmt_strs = [self.stringify(stmt) for stmt in node.get_children()]
        return "\n".join(stmt_strs)
    
    def stringify_StatementList(self, node: StatementList) -> str:
        """处理语句列表"""

        statements = []
        for stmt in node.get_children():
            stmt_str = self.stringify(stmt)
            if stmt_str:
                statements.append(stmt_str)
        
        # 如果没有有效语句，返回pass
        if not statements:
            return "pass"
        
        return "\n".join(statements)
    
    
    
    def stringify_ExpressionList(self, node: ExpressionList) -> str:
        """处理表达式列表"""
        if not node or not node.get_children():
            return "NoneExpr"
        
        exprs = []
        for expr in node.get_children():
            expr_str = self.stringify(expr)
            if expr_str:
                exprs.append(expr_str)
        
        # 如果没有有效语句，返回None
        if not exprs:
            return "NoneExpr"
        
        return "\n".join(exprs)
        
    def stringify_BinaryExpression(self, node: BinaryExpression) -> str:
        """处理二元表达式，包括字符串拼接"""
        # 特殊处理字符串拼接
        if node.op.value == "+" and self._is_string_literal(node.left) and self._is_string_literal(node.right):
            # 获取左右两边的字符串值
            left_str = self.stringify(node.left)
            right_str = self.stringify(node.right)
            
            # 如果两边都是字符串字面量，直接拼接它们（去掉右侧字符串的引号）
            if (left_str.startswith("'") and left_str.endswith("'") and 
                right_str.startswith("'") and right_str.endswith("'")):
                # 去掉右侧字符串的引号，直接拼接
                return f"{left_str} {right_str}"
        
        # 处理其他二元表达式
        if node.op.value == "&&":
            op = "and"
        elif node.op.value == "||":
            op = "or"
        else:
            op = node.op.value
        
        return f"{self.stringify(node.left)} {op} {self.stringify(node.right)}"

    def _is_string_literal(self, node) -> bool:
        """判断节点是否为字符串字面量"""
        if node.node_type == NodeType.LITERAL:
            value = node.value if hasattr(node, 'value') else None
            # 检查是否是字符串值
            if isinstance(value, str):
                # 检查是否被引号包围
                return (value.startswith("'") and value.endswith("'")) or \
                    (value.startswith('"') and value.endswith('"'))
        return False
    
    def stringify_ExpressionStatement(self, node: ExpressionStatement) -> str:
        """处理表达式语句"""
        if hasattr(node, 'expr') and node.expr:
            return self.stringify(node.expr)
        return ""
        
    def stringify_CallExpression(self, node: CallExpression) -> str:
        """
        将CallExpression对象转换为字符串
        """
        if not hasattr(node, 'callee') or not hasattr(node, 'args'):
            return ""
        
        callee_str = self.stringify(node.callee)
        
        # 处理参数
        arg_strs = []
        has_spread = False
        spread_args = []
        
        if hasattr(node, 'args') and node.args:
            for arg in node.args.get_children():
                # 检查是否有SpreadElement类型的参数
                if hasattr(arg, 'type') and arg.type == NodeType.SPREAD_ELEMENT:
                    has_spread = True
                    if hasattr(arg, 'argument'):
                        spread_arg = self.stringify(arg.argument)
                        if spread_arg:
                            spread_args.append(f"*{spread_arg}")
                else:
                    arg_str = self.stringify(arg)
                    if arg_str:
                        # 检查是否是关键字参数
                        if '=' in arg_str and not (arg_str.startswith("'") or arg_str.startswith('"')):
                            arg_strs.append(arg_str)
                        else:
                            arg_strs.append(arg_str)
        
        # 添加展开参数
        if spread_args:
            arg_strs.extend(spread_args)
        
        # 如果有SpreadParameter类型的参数，需要保留*前缀
        if hasattr(node, 'spread_args') and node.spread_args:
            for spread_arg in node.spread_args:
                spread_arg_str = self.stringify(spread_arg)
                if spread_arg_str:
                    arg_strs.append(spread_arg_str)
        
        return f"{callee_str}({', '.join(arg_strs)})"
    
    
    def stringify_FieldAccess(self, node: FieldAccess) -> str:
        """
        将FieldAccess对象转换为字符串
        """
        if node.object is None:
            return f".{self.stringify(node.field)}" if node.field else ""
        
        obj_str = self.stringify(node.object)
        field_str = self.stringify(node.field) if node.field else ""
        
        # 处理可能的空对象或字段
        if not obj_str:
            return field_str
        if not field_str:
            return obj_str
        
        return f"{obj_str}.{field_str}"
    
    def stringify_ArrayExpression(self, node: ArrayExpression) -> str:
        """处理数组表达式，包括列表、元组、集合、字典和连接字符串"""
        if not hasattr(node, 'elements') or not node.elements:
            return "[]"  # 默认返回空列表
        
        elements_str = []
        for element in node.elements.get_children():
            element_str = self.stringify(element)
            if element_str:
                elements_str.append(element_str)
        
        # 根据 is_pattern 的值选择不同的格式化方式
        if node.is_pattern == ArrayPatternType.LIST:
            return f"[{', '.join(elements_str)}]"
        elif node.is_pattern == ArrayPatternType.TUPLE:
            # 处理空元组和单元素元组的特殊情况
            if not elements_str:
                return "()"
            elif len(elements_str) == 1:
                return f"({elements_str[0]},)"
            else:
                return f"({', '.join(elements_str)})"
        elif node.is_pattern == ArrayPatternType.SET:
            if not elements_str:
                return "set()"  # 空集合需要使用 set() 函数
            return f"{{{', '.join(elements_str)}}}"
        elif node.is_pattern == ArrayPatternType.DICT:
            return f"{{{', '.join(elements_str)}}}"
        elif node.is_pattern == ArrayPatternType.PATTERN:
            # 用于模式匹配的表达式
            return f"{', '.join(elements_str)}"
        elif node.is_pattern == ArrayPatternType.CONCATENATED_STRING:
            # 处理连接字符串
            return ' '.join(elements_str)
        elif node.is_pattern == ArrayPatternType.SLICE:
            # 处理切片表达式
            return ':'.join(elements_str)
        elif node.is_pattern == ArrayPatternType.COMPREHENSION:
            # 处理推导式
            if hasattr(node, 'comprehension_expr') and node.comprehension_expr:
                comp_expr = self.stringify(node.comprehension_expr)
                return f"[{elements_str[0]} {comp_expr}]"
            return f"[{', '.join(elements_str)}]"
        else:
            # 默认作为列表处理
            return f"[{', '.join(elements_str)}]"
    
    def stringify_ArrayAccess(self, node: ArrayAccess) -> str:
        """
        将ArrayAccess对象转换为字符串，支持Python的各种切片操作
        包括基本索引、切片[start:end:step]、省略参数的切片[:end]、[start:]、[::step]等
        以及省略号与其他索引的组合，如[..., 0]、[..., :, tf.newaxis]等
        """
        if not node or not hasattr(node, 'array') or not hasattr(node, 'index'):
            return ""
        
        array_str = self.stringify(node.array) if node.array else ""
        
        # 处理索引表达式
        if hasattr(node.index, 'node_type'):
            # 处理数组表达式（可能是切片或复合索引）
            if node.index.node_type == NodeType.ARRAY_EXPR and hasattr(node.index, 'elements') and node.index.elements:
                elements = node.index.elements.get_children()
                
                # 检查是否是复合索引（包含多个元素，可能是切片、省略号或其他表达式的组合）
                if len(elements) > 0:
                    # 首先检查是否是简单的切片操作 [start:end:step]
                    is_simple_slice = True
                    for i, element in enumerate(elements):
                        # 如果有超过3个元素，或者有不是None/数字的元素，则不是简单切片
                        if i >= 3 or (isinstance(element, Literal) and element.value == "Ellipsis"):
                            is_simple_slice = False
                            break
                    
                    # 如果是简单切片，使用冒号语法
                    if is_simple_slice and len(elements) <= 3:
                        slice_parts = []
                        
                        # 处理start
                        if len(elements) > 0:
                            start = elements[0]
                            if isinstance(start, Literal) and start.value == "None":
                                slice_parts.append("")  # 省略start
                            else:
                                slice_parts.append(self.stringify(start))
                        
                        # 处理end
                        if len(elements) > 1:
                            end = elements[1]
                            if isinstance(end, Literal) and end.value == "None":
                                slice_parts.append("")  # 省略end
                            else:
                                slice_parts.append(self.stringify(end))
                        
                        # 处理step
                        if len(elements) > 2:
                            step = elements[2]
                            if isinstance(step, Literal) and step.value == "None":
                                slice_parts.append("")  # 省略step
                            else:
                                slice_parts.append(self.stringify(step))
                        
                        # 构建切片表达式
                        slice_expr = ":".join(slice_parts)
                        return f"{array_str}[{slice_expr}]"
                    
                    # 否则，处理为复合索引，使用逗号分隔
                    else:
                        index_parts = []
                        for element in elements:
                            if isinstance(element, Literal):
                                if element.value == "Ellipsis":
                                    index_parts.append("...")
                                elif element.value == "None":
                                    index_parts.append(":")  # 将None转换为:，表示完整切片
                                else:
                                    index_parts.append(self.stringify(element))
                            elif isinstance(element, ExpressionList):
                                # 处理嵌套的切片表达式
                                nested_elements = element.get_children()
                                if len(nested_elements) <= 3:  # 简单切片
                                    slice_parts = []
                                    for i, nested_elem in enumerate(nested_elements):
                                        if isinstance(nested_elem, Literal) and nested_elem.value == "None":
                                            slice_parts.append("")
                                        else:
                                            slice_parts.append(self.stringify(nested_elem))
                                    index_parts.append(":".join(slice_parts))
                                else:
                                    # 复杂嵌套表达式
                                    index_parts.append(self.stringify(element))
                            else:
                                # 其他表达式（如字段访问、函数调用等）
                                index_parts.append(self.stringify(element))
                        
                        return f"{array_str}[{', '.join(index_parts)}]"
                
                # 空数组表达式
                return f"{array_str}[]"
            
            # 处理其他类型的索引表达式
            elif node.index.node_type == NodeType.EXPRESSION_LIST:
                # 处理表达式列表作为索引
                elements = node.index.get_children()
                index_parts = []
                
                for element in elements:
                    if isinstance(element, Literal) and element.value == "Ellipsis":
                        index_parts.append("...")
                    else:
                        index_parts.append(self.stringify(element))
                
                return f"{array_str}[{', '.join(index_parts)}]"
        
        # 处理普通索引访问
        index_str = self.stringify(node.index)
        
        # 处理特殊字面量
        if isinstance(node.index, Literal):
            value = node.index.value
            if isinstance(value, str):
                # 处理负数索引
                if value.startswith('-') and value[1:].isdigit():
                    index_str = value
                # 处理正数索引
                elif value.isdigit():
                    index_str = value
                # 处理省略号
                elif value == "Ellipsis":
                    index_str = "..."
        
        return f"{array_str}[{index_str}]"
    
    
    def stringify_Literal(self, node: Literal) -> str:
        """处理字面量，包括字符串、数字、布尔值等"""
        if not hasattr(node, 'value'):
            return "None"
        
        # 检查是否有明确的字符串标记
        is_string = hasattr(node, 'is_string') and node.is_string
        
        # 如果有原始文本，优先使用
        if hasattr(node, 'raw_text') and node.raw_text:
            return node.raw_text
        
        # 处理字符串字面量
        if is_string:
            # 检查引号类型
            quote_type = getattr(node, 'quote_type', 'single')
            
            if quote_type == "double":
                return f'"{node.value}"'
            elif quote_type == "triple_single":
                return f"'''{node.value}'''"
            elif quote_type == "triple_double":
                return f'"""{node.value}"""'
            else:  # 默认使用单引号
                return f"'{node.value}'"
        
        # 处理特殊字面量
        if node.value == "True" or node.value == "False" or node.value == "None":
            return node.value
        
        # 检查是否是数字
        try:
            # 尝试将值解析为整数
            int(node.value)
            return node.value
        except ValueError:
            try:
                # 尝试将值解析为浮点数
                float(node.value)
                return node.value
            except ValueError:
                # 不是数字，当作字符串处理
                # 检查值是否已经带有引号
                if (node.value.startswith("'") and node.value.endswith("'")) or \
                (node.value.startswith('"') and node.value.endswith('"')):
                    return node.value
                else:
                    # 默认添加单引号
                    return f"'{node.value}'"

    
    def stringify_ForInStatement(self, node: ForInStatement) -> str:
        """
        将ForInStatement对象转换为字符串
        """
        # 获取迭代变量
        var_str = self.stringify(node.declarator) if node.declarator else "_"
        
        # 获取可迭代对象
        iterable_str = self.stringify(node.iterable) if node.iterable else "[]"
        
        # 处理循环体
        body_str = self.stringify(node.body) if node.body else "pass"
        
        # 如果循环体不是以换行开始，添加换行和缩进
        if body_str and not body_str.startswith("\n"):
            body_str = "\n" + self._indent(body_str)
        
        # 处理异步for循环
        async_prefix = "async " if node.is_async else ""
        
        return f"{async_prefix}for {var_str} in {iterable_str}:{body_str}"
    
    def stringify_ReturnStatement(self, node: ReturnStatement) -> str:
        """
        将ReturnStatement对象转换为字符串
        """
        if node.expr is None:
            return "return"
        
        expr_str = self.stringify(node.expr)
        return f"return {expr_str}"
    
    def stringify_IfStatement(self, node: IfStatement) -> str:
        """处理if语句"""
        if not hasattr(node, 'condition') or not hasattr(node, 'consequence'):
            return ""
        
        condition = self.stringify(node.condition)
        consequence = self.stringify(node.consequence)
        
        # 处理空的consequence块
        #if not consequence:
        #    consequence = "pass"
        
        result = f"if {condition}:\n{self._indent(consequence)}"
        
        # 处理else部分
        if hasattr(node, 'alternate') and node.alternate:
            alternate = self.stringify(node.alternate)
            
            # 处理空的alternative块
            if isinstance(node.alternate, BlockStatement) and (not alternate or alternate.strip() == ""):
                alternate = "pass"
            
            # 检查是否是elif (在内部表示中是嵌套的if)
            if isinstance(node.alternate, IfStatement):
                # 移除缩进并去掉开头的"if"
                alternate_lines = alternate.split('\n')
                if alternate_lines[0].startswith("if "):
                    alternate_lines[0] = "el" + alternate_lines[0]
                    alternative = '\n'.join(alternate_lines)
                    result += f"\n{alternative}"
                else:
                    result += f"\nelse:\n{self._indent(alternate)}"
            else:
                result += f"\nelse:\n{self._indent(alternate)}"
        
        return result
    
    def stringify_WhileStatement(self, node: WhileStatement) -> str:
        """处理while语句"""
        if not hasattr(node, 'condition') or not hasattr(node, 'body'):
            return ""
        
        condition = self.stringify(node.condition)
        body = self.stringify(node.body)
        
        return f"while {condition}:\n{self._indent(body)}"
    
    def stringify_TryStatement(self, node: TryStatement) -> str:
        """处理try语句"""
        if not hasattr(node, 'body'):
            return ""
        
        body = self.stringify(node.body)
        result = f"try:\n{self._indent(body)}"
        
        # 处理except块
        if hasattr(node, 'handlers') and node.handlers:
            for handler in node.handlers.node_list:
                # 获取异常类型
                exception_type = ""
                if hasattr(handler, 'catch_types') and handler.catch_types:
                    exception_type = self.stringify(handler.catch_types)
                
                # 获取异常变量
                exception_var = ""
                if hasattr(handler, 'exception') and handler.exception:
                    exception_var = f" as {self.stringify(handler.exception)}"
                
                # 获取except块的主体
                handler_body = self.stringify(handler.body) if handler.body else "pass"
                
                # 构建except子句
                if exception_type:
                    result += f"\nexcept {exception_type}{exception_var}:\n{self._indent(handler_body)}"
                else:
                    result += f"\nexcept{exception_var}:\n{self._indent(handler_body)}"
        
        # 处理else块
        if hasattr(node, 'else_block') and node.else_block:
            else_body = self.stringify(node.else_block)
            result += f"\nelse:\n{self._indent(else_body)}"
        
        # 处理finally块
        if hasattr(node, 'finalizer') and node.finalizer:
            finally_body = self.stringify(node.finalizer.body)
            result += f"\nfinally:\n{self._indent(finally_body)}"
        elif hasattr(node, 'handlers') and node.handlers and hasattr(node.handlers, 'finally_block') and node.handlers.finally_block:
            finally_body = self.stringify(node.handlers.finally_block)
            result += f"\nfinally:\n{self._indent(finally_body)}"
        
        return result
    
    def stringify_AssertStatement(self, node) -> str:
        """处理assert语句"""
        if not hasattr(node, 'condition'):
            return "assert True"
        
        condition = self.stringify(node.condition)
        
        if hasattr(node, 'message') and node.message:
            message = self.stringify(node.message)
            return f"assert {condition}, {message}"
        
        return f"assert {condition}"
    
    def stringify_RaiseStatement(self, node) -> str:
        """
        处理raise语句
        
        支持以下形式:
        1. raise - 不带参数的重新抛出异常
        2. raise Exception - 抛出指定异常
        3. raise Exception("message") - 带消息的异常
        4. raise Exception from cause - 带原因的异常
        """
        # 处理不带参数的raise语句
        if not hasattr(node, 'expression') or node.expression is None:
            return "raise"
        
        expression = self.stringify(node.expression)
        
        # 处理带from子句的raise语句
        if hasattr(node, 'cause') and node.cause is not None:
            cause = self.stringify(node.cause)
            return f"raise {expression} from {cause}"
        
        # 处理普通的带异常的raise语句
        return f"raise {expression}"
    
    def stringify_WithStatement(self, node: WithStatement) -> str:
        """
        将WithStatement对象转换为字符串
        
        支持以下形式:
        1. 单个资源: with open('file.txt') as f:
        2. 多个资源: with open('file1.txt') as f1, open('file2.txt') as f2:
        3. 不带as的资源: with lock:
        4. 带异步前缀: async with session.get(url) as response:
        """
        if not hasattr(node, 'object') or not hasattr(node, 'body'):
            return ""
        
        # 处理with语句的资源部分
        resources = []
        
        # 检查是否是多个资源（数组表达式）
        if hasattr(node.object, 'node_type') and node.object.node_type == NodeType.ARRAY_EXPR and hasattr(node.object, 'elements'):
            # 处理多个资源
            for element in node.object.elements.get_children():
                resource_str = self._process_with_resource(element)
                if resource_str:
                    resources.append(resource_str)
        else:
            # 单个资源
            resource_str = self._process_with_resource(node.object)
            if resource_str:
                resources.append(resource_str)
        
        # 检查是否是异步with语句
        async_prefix = "async " if hasattr(node, 'is_async') and node.is_async else ""
        
        # 构建with语句的头部
        with_header = f"{async_prefix}with " + ", ".join(resources)
        
        # 处理with语句的主体
        body = self.stringify(node.body) if node.body else "pass"
        
        # 如果主体为空，添加pass语句
        if not body or body.strip() == "":
            body = "pass"
        
        # 返回完整的with语句
        return f"{with_header}:\n{self._indent(body)}"

    def _process_with_resource(self, resource) -> str:
        """处理with语句中的资源表达式，支持as模式"""
        if not resource:
            return ""
        
        # 检查是否是带as的资源（使用FieldAccess表示）
        if hasattr(resource, 'node_type') and resource.node_type == NodeType.FIELD_ACCESS:
            # 检查是否是as模式
            if hasattr(resource, 'is_as_pattern') and resource.is_as_pattern:
                obj_str = self.stringify(resource.object)
                field_str = self.stringify(resource.field)
                return f"{obj_str} as {field_str}"
        
        # 普通资源表达式
        return self.stringify(resource)
    
    def stringify_PassStatement(self, node) -> str:
        """处理pass语句"""
        return "pass"
    
    def stringify_BreakStatement(self, node) -> str:
        """处理break语句"""
        return "break"
    
    def stringify_ContinueStatement(self, node) -> str:
        """处理continue语句"""
        return "continue"
    
    def stringify_SpreadElement(self, node) -> str:
        """处理函数调用中的展开元素，如 *args 或 **kwargs"""

        
        arg_str = self.stringify(node.expr)
        
        # 判断是否为字典展开
        is_dict_spread = False
        if hasattr(node, 'is_dict'):
            is_dict_spread = node.is_dict
        
        prefix = "**" if is_dict_spread else "*"
        return f"{prefix}{arg_str}"
    
    def stringify_KeywordArgument(self, node) -> str:
        """处理关键字参数"""
        if not hasattr(node, 'name') or not hasattr(node, 'value'):
            return ""
        
        name = self.stringify(node.name)
        value = self.stringify(node.value)
        
        return f"{name}={value}"
    
    def stringify_DefaultParameter(self, node) -> str:
        """处理默认参数"""
        if not hasattr(node, 'name') or not hasattr(node, 'value'):
            return ""
        
        name = self.stringify(node.name)
        value = self.stringify(node.value)
        
        return f"{name}={value}"
    
    def stringify_DictionaryLiteral(self, node) -> str:
        """处理字典字面量"""
        if not hasattr(node, 'entries'):
            return "{}"
        
        entries = []
        for entry in node.entries.get_children():
            entries.append(self.stringify(entry))
        
        return f"{{{', '.join(entries)}}}"
    
    def stringify_DictionaryEntry(self, node) -> str:
        """处理字典条目"""
        if not hasattr(node, 'key') or not hasattr(node, 'value'):
            return ""
        
        key = self.stringify(node.key)
        value = self.stringify(node.value)
        
        return f"{key}: {value}"
    
    def stringify_ComprehensionExpression(self, node) -> str:
        """将ComprehensionExpression转换为字符串"""
        if not hasattr(node, 'body') or not node.body:
            return "[]"  # 默认返回空列表
        
        body_str = self.stringify(node.body)
        
        # 构建子句字符串
        clauses_str = []
        for clause in node.clauses:
            if clause.is_for:
                left_str = self.stringify(clause.left) if clause.left else "_"
                right_str = self.stringify(clause.right) if clause.right else "range(0)"
                clauses_str.append(f"for {left_str} in {right_str}")
            else:
                condition_str = self.stringify(clause.condition) if clause.condition else "True"
                clauses_str.append(f"if {condition_str}")
        
        # 根据推导式类型选择不同的括号
        if node.comprehension_type == "list":
            return f"[{body_str} {' '.join(clauses_str)}]"
        elif node.comprehension_type == "dict":
            return f"{{{body_str} {' '.join(clauses_str)}}}"
        elif node.comprehension_type == "set":
            return f"{{{body_str} {' '.join(clauses_str)}}}"
        elif node.comprehension_type == "generator":
            return f"({body_str} {' '.join(clauses_str)})"
        else:
            return f"[{body_str} {' '.join(clauses_str)}]"  # 默认使用列表语法
        
        
    def stringify_UnaryExpression(self, node: UnaryExpression) -> str:
        from mutable_tree.nodes import UnaryOps
        op = node.op
        if op in {UnaryOps.TYPEOF, UnaryOps.DELETE}:
            return f"{op.value} {self.stringify(node.operand)}"
        elif op == UnaryOps.NOT:
            op.value = "not"
            return f"{op.value} {self.stringify(node.operand)}"
        else:
            return f"{op.value} {self.stringify(node.operand)}"

    def stringify_LambdaExpression(self, node) -> str:
        """处理Lambda表达式"""
        if not hasattr(node, 'params') or not hasattr(node, 'body'):
            return "lambda: None"
        
        # 处理参数
        params = []
        if node.params and hasattr(node.params, 'get_children'):
            for param in node.params.get_children():
                param_str = self.stringify(param)
                if param_str:
                    params.append(param_str)
        
        params_str = ", ".join(params)
        
        # 处理函数体
        body_str = self.stringify(node.body) if node.body else "None"
        
        return f"lambda {params_str}: {body_str}"
