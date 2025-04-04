from ..node import NodeType, NodeList
from .statement import Statement
from .statement import is_statement
from ..utils import throw_invalid_type
from typing import List


class StatementList(NodeList):
    node_list: List[Statement]

    def __init__(self, node_type: NodeType, stmts: List[Statement]):
        super().__init__(node_type)
        self.node_list = stmts
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.STATEMENT_LIST:
            throw_invalid_type(self.node_type, self)

        for i, stmt in enumerate(self.node_list):
            if not is_statement(stmt):
                throw_invalid_type(stmt.node_type, self, f"stmt#{i}")
                
    def get_children(self) -> List[Statement]:
        """返回语句列表中的所有语句"""
        return self.node_list
    
    def get_children_names(self) -> List[int]:
        """返回语句列表中所有语句的索引"""
        return list(range(len(self.node_list)))
