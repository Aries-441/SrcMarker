from .common import BaseStringifier
from mutable_tree.nodes import NodeType
from mutable_tree.nodes.statements.declarators import Declarator
from mutable_tree.nodes.statements.func_declaration import FunctionHeader, FunctionDeclaration, FormalParameterList
from mutable_tree.nodes import Statement, Expression, BlockStatement, StatementList
from mutable_tree.nodes import Identifier, Literal


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
        decl_str = self.stringify(node.func_decl)
        
        # 获取参数列表
        params = []
        if hasattr(node.func_decl, 'parameters') and node.func_decl.parameters is not None:
            # 直接从原始代码中获取参数列表
            if hasattr(node, '_original_params') and node._original_params:
                return f"{decl_str}({node._original_params})"
            
            # 如果没有原始参数，尝试从参数列表构建
            for param in node.func_decl.parameters.get_children():
                params.append(self.stringify(param))
        
        # 构建函数头字符串
        return f"{decl_str}({', '.join(params)})"
    
    def stringify_FunctionDeclaration(self, node: FunctionDeclaration) -> str:
        """
        将FunctionDeclaration对象转换为字符串
        """
        # 如果有原始文本，直接使用
        if hasattr(node, '_original_text') and node._original_text:
            return node._original_text
        
        header_str = self.stringify(node.header)
        
        # 如果有原始函数体，直接使用
        if hasattr(node, '_original_body') and node._original_body:
            return f"def {header_str}:\n{self._indent(node._original_body)}"
        
        # 处理函数体
        if node.body and isinstance(node.body, BlockStatement):
            # 获取函数体中的语句列表
            stmt_list = node.body.statements if hasattr(node.body, 'statements') else None
            if stmt_list and isinstance(stmt_list, StatementList):
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
        
        # Python函数定义格式
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
        return node.value
    
    def _indent(self, text: str, spaces: int = 4) -> str:
        """
        缩进文本
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        indented_lines = [' ' * spaces + line for line in lines]
        return '\n'.join(indented_lines)