from ..utils import throw_invalid_type
from typing import List, Optional
from mutable_tree.nodes.node import Node
from ..node import NodeType
from .expression import Expression
from .expression import is_primary_expression, is_expression

class ComprehensionClause(Node):
    """表示推导式中的子句（for或if）"""
    def __init__(self, node_type: NodeType, is_for: bool, left=None, right=None, condition=None):
        super().__init__(node_type)
        self.is_for = is_for  # True表示for子句，False表示if子句
        self.left = left      # for子句中的变量
        self.right = right    # for子句中的可迭代对象
        self.condition = condition  # if子句中的条件
    
    def get_children(self) -> List[Node]:
        if self.is_for:
            return [self.left, self.right]
        else:
            return [self.condition]
    
    def get_children_names(self) -> List[str]:
        if self.is_for:
            return ["left", "right"]
        else:
            return ["condition"]


class ComprehensionExpression(Expression):
    """表示列表、集合、字典推导式"""
    def __init__(
        self, 
        node_type: NodeType, 
        body: Expression, 
        clauses: List[ComprehensionClause],
        comprehension_type: str = "list"  # 可以是 "list", "dict", "set", "generator"
    ):
        super().__init__(node_type)
        self.body = body
        self.clauses = clauses
        self.comprehension_type = comprehension_type
        self._check_types()
    
    def _check_types(self):
        if self.node_type != NodeType.COMPREHENSION_EXPR:
            throw_invalid_type(self.node_type, self)
    
    def get_children(self) -> List[Node]:
        return [self.body] + self.clauses
    
    def get_children_names(self) -> List[str]:
        return ["body"] + [f"clause_{i}" for i in range(len(self.clauses))]