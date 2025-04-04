'''
FileName: 
Description: 
Autor: Liujunjie/Aries-441
Github: https://github.com/Aries-441
Date: 2025-03-19 19:52:00
E-mail: sjtu.liu.jj@gmail.com/sjtu.1518228705@sjtu.edu.cn
LastEditTime: 2025-03-19 19:52:14
'''
from ..node import Node, NodeType
from .statement import Statement
from ..expressions import Expression
from ..expressions import is_expression
from ..utils import throw_invalid_type
from typing import List, Optional


class RaiseStatement(Statement):
    def __init__(
        self,
        node_type: NodeType,
        expression: Optional[Expression] = None,
        cause: Optional[Expression] = None,
    ):
        super().__init__(node_type)
        self.expression = expression
        self.cause = cause
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.RAISE_STMT:
            throw_invalid_type(self.node_type, self)
        if self.expression is not None and not is_expression(self.expression):
            throw_invalid_type(self.expression.node_type, self, attr="expression")
        if self.cause is not None and not is_expression(self.cause):
            throw_invalid_type(self.cause.node_type, self, attr="cause")

    def get_children(self) -> List[Node]:
        children = []
        if self.expression is not None:
            children.append(self.expression)
        if self.cause is not None:
            children.append(self.cause)
        return children

    def get_children_names(self) -> List[str]:
        return ["expression", "cause"]