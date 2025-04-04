from ..node import Node, NodeType
from .statement import Statement
from ..expressions import Expression, is_expression
from ..utils import throw_invalid_type
from typing import List, Optional


class ClassDeclaration(Statement):
    def __init__(
        self,
        node_type: NodeType,
        name: Expression,
        body: Statement,
        superclasses: Optional[List[Expression]] = None,
    ):
        super().__init__(node_type)
        self.name = name
        self.body = body
        self.superclasses = superclasses or []
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.CLASS_DECLARATION:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.name):
            throw_invalid_type(self.name.node_type, self, attr="name")
        if not isinstance(self.body, Statement):
            throw_invalid_type(self.body.node_type, self, attr="body")
        for i, superclass in enumerate(self.superclasses):
            if not is_expression(superclass):
                throw_invalid_type(
                    superclass.node_type, self, attr=f"superclasses[{i}]"
                )

    def get_children(self) -> List[Node]:
        children = [self.name, self.body]
        children.extend(self.superclasses)
        return children

    def get_children_names(self) -> List[str]:
        names = ["name", "body"]
        names.extend([f"superclass_{i}" for i in range(len(self.superclasses))])
        return names