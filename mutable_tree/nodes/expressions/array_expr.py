'''
FileName: 
Description: 
Autor: Liujunjie/Aries-441
Github: https://github.com/Aries-441
Date: 2025-03-19 19:00:14
E-mail: sjtu.liu.jj@gmail.com/sjtu.1518228705@sjtu.edu.cn
LastEditTime: 2025-04-07 14:24:05
'''
from ..node import Node, NodeType
from .expression import Expression
from .expression_list import ExpressionList
from ..utils import throw_invalid_type
from typing import List, Optional
from enum import IntEnum


class ArrayPatternType(IntEnum):
    """Python 数据结构类型枚举"""
    LIST = 0        # 列表 [1, 2, 3]
    TUPLE = 1       # 元组 (1, 2, 3)
    SET = 2         # 集合 {1, 2, 3}
    DICT = 3        # 字典 {key: value}
    PATTERN = 4     # 模式匹配 case [a, b, c]:
    CONCATENATED_STRING = 5  # 连接字符串 f"hello {name}"
    SLICE = 6       # 切片表达式 arr[1:10:2]
    COMPREHENSION = 7  # 推导式 [x for x in range(10)]


class ArrayExpression(Expression):
    """
    表示 Python 中的各种集合类型数据结构，如列表、元组、集合、字典等
    is_pattern: 使用 ArrayPatternType 枚举值区分不同的数据结构类型
    """
    def __init__(self, node_type: NodeType, elements: ExpressionList, 
                 is_pattern: int = ArrayPatternType.LIST,
                 is_comprehension: bool = False,
                 comprehension_expr: Optional[Expression] = None):
        super().__init__(node_type)
        self.elements = elements
        self._check_types()
        self.is_pattern = is_pattern
        self.is_comprehension = is_comprehension
        self.comprehension_expr = comprehension_expr

    def _check_types(self):
        if self.node_type != NodeType.ARRAY_EXPR:
            throw_invalid_type(self.node_type, self)
        if self.elements.node_type != NodeType.EXPRESSION_LIST:
            throw_invalid_type(self.elements.node_type, self, "elements")

    def get_children(self) -> List[Node]:
        children = [self.elements]
        if self.comprehension_expr:
            children.append(self.comprehension_expr)
        return children

    def get_children_names(self) -> List[str]:
        names = ["elements"]
        if self.comprehension_expr:
            names.append("comprehension_expr")
        return names
