from .common import BaseStringifier
from mutable_tree.nodes import NodeType
from mutable_tree.nodes.statements.declarators import Declarator
from mutable_tree.nodes.statements.func_declaration import FunctionHeader, FunctionDeclaration, FormalParameterList
from mutable_tree.nodes import Statement, Expression, BlockStatement, StatementList
from mutable_tree.nodes import Identifier, Literal
from mutable_tree.nodes.expressions.binary_expr import BinaryExpression
from mutable_tree.nodes.expressions.unary_expr import UnaryExpression
from mutable_tree.nodes.expressions.call_expr import CallExpression
from mutable_tree.nodes.expressions.field_access import FieldAccess as MemberExpression
from mutable_tree.nodes.statements.if_stmt import IfStatement
from mutable_tree.nodes.statements.for_in_stmt import ForInStatement
from mutable_tree.nodes.statements.while_stmt import WhileStatement
from mutable_tree.nodes.statements.try_stmt import TryStatement


class PythonStringifier(BaseStringifier):
    def __init__(self):
        super().__init__()
    
    def stringify_Declarator(self, node: Declarator) -> str:
        """
        将Declarator对象转换为字符串
        """
        # 对于Python函数声明中的Declarator，我们只需要返回标识符名称
        if hasattr(node, 'declarator') and node.declarator is not None:
            return self.stringify(node.declarator)
        return ""
        
    def stringify_FunctionHeader(self, node: FunctionHeader) -> str:
        """
        将FunctionHeader对象转换为字符串
        """
        # 获取函数名
        if hasattr(node, 'func_decl') and node.func_decl is not None:
            # 从func_decl中提取函数名
            if hasattr(node.func_decl, 'declarator') and node.func_decl.declarator is not None:
                if hasattr(node.func_decl.declarator, 'decl_id') and node.func_decl.declarator.decl_id is not None:
                    func_name = self.stringify(node.func_decl.declarator.decl_id)
                else:
                    # 如果找不到decl_id，尝试直接获取declarator
                    func_name = self.stringify(node.func_decl.declarator)
            else:
                # 如果找不到declarator，尝试直接获取func_decl
                func_name = self.stringify(node.func_decl)
        else:
            # 如果找不到func_decl，返回空函数名
            func_name = ""
        
        # 获取参数列表
        params = []
        if hasattr(node.func_decl, 'parameters') and node.func_decl.parameters is not None:
            # 从参数列表构建
            for param in node.func_decl.parameters.get_children():
                param_str = self.stringify(param)
                if param_str:
                    params.append(param_str)
        
        # 构建函数头字符串
        return f"{func_name}({', '.join(params)})"
    
    def stringify_FunctionDeclaration(self, node: FunctionDeclaration) -> str:
        """
        将FunctionDeclaration对象转换为字符串
        """
        header_str = self.stringify(node.header)
        
        # 处理函数体
        if node.body and isinstance(node.body, BlockStatement):
            # 获取函数体中的语句列表
            stmt_list = node.body.stmts if hasattr(node.body, 'stmts') else None
            if stmt_list and isinstance(stmt_list, StatementList):
                # 修复：使用get_children()方法获取语句列表，而不是尝试访问不存在的statements属性
                statements = stmt_list.get_children()
                
                # 处理函数体内容
                body_lines = []
                for stmt in statements:
                    stmt_str = self.stringify(stmt)
                    if stmt_str:
                        body_lines.append(stmt_str)
                
                # 如果函数体为空，添加pass语句
                if not body_lines:
                    body_lines.append("pass")
                
                body_str = "\n".join(body_lines)
            else:
                body_str = "pass"
        else:
            body_str = "pass"
        
            return f"def {header_str}:\n{self._indent(body_str)}"
    
    def stringify_UntypedParameter(self, node) -> str:
        """
        处理无类型参数
        """
        if hasattr(node, 'declarator') and node.declarator is not None:
            param_name = self.stringify(node.declarator)
            
            # 处理默认值
            if hasattr(node, 'default_value') and node.default_value is not None:
                default_value = self.stringify(node.default_value)
                return f"{param_name}={default_value}"
            
            return param_name
        return ""
    
    # 添加对文档字符串的支持
    def stringify_DocString(self, node) -> str:
        """
        处理文档字符串
        """
        if hasattr(node, 'value'):
            return f'"""{node.value}"""'
        return ""
    
    def stringify_VariableDeclarator(self, node) -> str:
        """
        处理变量声明器
        """
        name = self.stringify(node.name) if hasattr(node, 'name') and node.name else ""
        initializer = ""
        
        if hasattr(node, 'initializer') and node.initializer:
            initializer = f" = {self.stringify(node.initializer)}"
        
        return f"{name}{initializer}"
    
    def stringify_ExpressionStatement(self, node) -> str:
        """
        处理表达式语句
        """
        if hasattr(node, 'expression') and node.expression:
            return self.stringify(node.expression)
        return ""
    
    def stringify_ReturnStatement(self, node) -> str:
        """
        处理return语句
        """
        if hasattr(node, 'expression') and node.expression:
            return f"return {self.stringify(node.expression)}"
        return "return"
    
    def stringify_Literal(self, node: Literal) -> str:
        """
        处理字面量
        """
        if not hasattr(node, 'value'):
            return ""
        
        if node.node_type == NodeType.LITERAL:
            # 根据值的类型决定如何格式化
            if isinstance(node.value, str):
                # 处理字符串字面量，添加引号
                # 检查是否已经有引号
                if node.value.startswith('"') and node.value.endswith('"'):
                    return node.value
                elif node.value.startswith("'") and node.value.endswith("'"):
                    return node.value
                else:
                    return f'"{node.value}"'
            elif node.value is None:
                # 处理None
                return "None"
            elif isinstance(node.value, bool):
                # 处理布尔值，Python中是True/False
                return str(node.value)
            else:
                # 处理数字等其他字面量
                return str(node.value)
        
        # 处理特定类型的字面量
        if hasattr(node, 'value'):
            return str(node.value)
        
        return ""
    
    def stringify_Identifier(self, node: Identifier) -> str:
        """
        处理标识符
        """
        return node.name
    
    def stringify_BinaryExpression(self, node: BinaryExpression) -> str:
        """
        处理二元表达式
        """
        left = self.stringify(node.left)
        right = self.stringify(node.right)
        
        # 修复：直接使用op字符串，而不是尝试访问其value属性
        op = node.op
        
        # 处理Python特有的操作符
        if op == "===":
            op = "=="
        elif op == "!==":
            op = "!="
        elif op == "&&":
            op = "and"
        elif op == "||":
            op = "or"
        elif op == "!":
            op = "not"
        
        return f"{left} {op} {right}"
    
    def stringify_UnaryExpression(self, node: UnaryExpression) -> str:
        """
        处理一元表达式
        """
        operand = self.stringify(node.operand)
        
        # 修复：直接使用op字符串，而不是尝试访问其value属性
        op = node.op
        
        # 处理Python特有的操作符
        if op == "!":
            op = "not "
            return f"{op}{operand}"
        
        return f"{op}{operand}"
    
    def stringify_ForInStatement(self, node: ForInStatement) -> str:
        """
        处理for-in语句
        """
        left = self.stringify(node.left)
        right = self.stringify(node.right)
        body = self.stringify(node.body)
        
        # 添加对forin_type的处理
        if hasattr(node, 'forin_type') and node.forin_type == "enumerate":
            return f"for {left} in enumerate({right}):\n{self._indent(body)}"
        else:
            return f"for {left} in {right}:\n{self._indent(body)}"
    
    def stringify_CallExpression(self, node: CallExpression) -> str:
        """
        处理函数调用表达式
        """
        callee = self.stringify(node.callee)
        args = []
        
        if node.arguments:
            for arg in node.arguments.get_children():
                args.append(self.stringify(arg))
        
        return f"{callee}({', '.join(args)})"
    
    def stringify_MemberExpression(self, node: MemberExpression) -> str:
        """
        处理成员表达式
        """
        obj = self.stringify(node.object)
        prop = self.stringify(node.property)
        
        # Python使用点号访问属性
        return f"{obj}.{prop}"
    
    def stringify_IfStatement(self, node: IfStatement) -> str:
        """
        处理if语句
        """
        condition = self.stringify(node.condition)
        consequence = self.stringify(node.consequence)
        
        # 处理if语句主体
        result = f"if {condition}:\n{self._indent(consequence)}"
        
        # 处理else部分 - 检查属性是否存在
        if hasattr(node, 'alternative') and node.alternative:
            alternative = self.stringify(node.alternative)
            
            # 检查是否是elif (在内部表示中是嵌套的if)
            if isinstance(node.alternative, IfStatement):
                # 移除缩进并去掉开头的"if"
                alternative_lines = alternative.split('\n')
                if alternative_lines[0].startswith("if "):
                    alternative_lines[0] = "el" + alternative_lines[0]
                    alternative = '\n'.join(alternative_lines)
                    result += f"\n{alternative}"
                else:
                    result += f"\nelse:\n{self._indent(alternative)}"
            else:
                result += f"\nelse:\n{self._indent(alternative)}"
        
        return result
    
    def stringify_WhileStatement(self, node: WhileStatement) -> str:
        """
        处理while语句
        """
        condition = self.stringify(node.condition)
        body = self.stringify(node.body)
        
        return f"while {condition}:\n{self._indent(body)}"
    
    def stringify_TryStatement(self, node: TryStatement) -> str:
        """
        处理try语句
        """
        body = self.stringify(node.body)
        result = f"try:\n{self._indent(body)}"
        
        # 处理catch块
        if node.handler and node.handler.body:
            handler_body = self.stringify(node.handler.body)
            param = ""
            if node.handler.param:
                param = f" as {self.stringify(node.handler.param)}"
            
            # Python中使用except而不是catch
            result += f"\nexcept Exception{param}:\n{self._indent(handler_body)}"
        
        # 处理finally块
        if node.finalizer:
            finalizer_body = self.stringify(node.finalizer)
            result += f"\nfinally:\n{self._indent(finalizer_body)}"
        
        return result
    
    def stringify_ImportStatement(self, node) -> str:
        """
        处理import语句
        """
        if hasattr(node, 'module') and node.module:
            module = self.stringify(node.module)
            
            # 处理导入的具体项
            if hasattr(node, 'names') and node.names:
                names = []
                for name in node.names:
                    names.append(self.stringify(name))
                
                if names:
                    return f"from {module} import {', '.join(names)}"
            
            return f"import {module}"
        return ""
    
    def stringify_ClassDeclaration(self, node) -> str:
        """
        处理类声明
        """
        name = self.stringify(node.name) if hasattr(node, 'name') and node.name else ""
        
        # 处理继承
        bases = []
        if hasattr(node, 'bases') and node.bases:
            for base in node.bases:
                bases.append(self.stringify(base))
        
        bases_str = f"({', '.join(bases)})" if bases else ""
        
        # 处理类体
        body = ""
        if hasattr(node, 'body') and node.body:
            body = self.stringify(node.body)
            if not body.strip():
                body = "pass"
        else:
            body = "pass"
        
        return f"class {name}{bases_str}:\n{self._indent(body)}"
    
    def _indent(self, text: str, spaces: int = 4) -> str:
        """
        缩进文本
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        indented_lines = [' ' * spaces + line for line in lines]
        return '\n'.join(indented_lines)