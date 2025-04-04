'''
FileName: 
Description: 
Autor: Liujunjie/Aries-441
Github: https://github.com/Aries-441
Date: 2025-03-19 20:00:01
E-mail: sjtu.liu.jj@gmail.com/sjtu.1518228705@sjtu.edu.cn
LastEditTime: 2025-03-19 20:00:35
'''
from ..node import Node, NodeType
from .statement import Statement
from ..utils import throw_invalid_type
from typing import List, Optional


class VariableDeclaration(Statement):
    def __init__(
        self,
        node_type: NodeType,
        declarators: List["VariableDeclarator"],
        type_name: Optional[str] = None,
    ):
        super().__init__(node_type)
        self.declarators = declarators
        self.type_name = type_name
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.VARIABLE_DECLARATION:
            throw_invalid_type(self.node_type, self)
        for i, declarator in enumerate(self.declarators):
            if not isinstance(declarator, VariableDeclarator):
                throw_invalid_type(
                    declarator.node_type, self, attr=f"declarators[{i}]"
                )

    def get_children(self) -> List[Node]:
        return self.declarators

    def get_children_names(self) -> List[str]:
        return [f"declarator_{i}" for i in range(len(self.declarators))]


class VariableDeclarator(Node):
    def __init__(
        self,
        name: "Expression",
        initializer: Optional["Expression"] = None,
        node_type: NodeType = NodeType.VARIABLE_DECLARATOR,
    ):
        super().__init__(node_type)
        self.name = name
        self.initializer = initializer
        self._check_types()

    def _check_types(self):
        from ..expressions import is_expression

        if self.node_type != NodeType.VARIABLE_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.name):
            throw_invalid_type(self.name.node_type, self, attr="name")
        if self.initializer is not None and not is_expression(self.initializer):
            throw_invalid_type(self.initializer.node_type, self, attr="initializer")

    def get_children(self) -> List[Node]:
        children = [self.name]
        if self.initializer is not None:
            children.append(self.initializer)
        return children

    def get_children_names(self) -> List[str]:
        names = ["name"]
        if self.initializer is not None:
            names.append("initializer")
        return names