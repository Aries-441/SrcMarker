from .visitor import TransformingVisitor
from mutable_tree.nodes import Node, NodeType
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import WhileStatement, ForInStatement
from typing import Optional


class WhileToForVisitor(TransformingVisitor):
    def __init__(self, lang="c"):
        super().__init__()
        self.lang = lang

    def visit_WhileStatement(
        self,
        node: WhileStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)
        new_stmts = []
        condition = node.condition
        body = node.body

        if self.lang == "python":
            # Python风格的for循环转换
            # 创建一个可迭代对象，通常是range()
            # 假设条件是 i < n 形式，转换为 for i in range(n)
            if condition.node_type == NodeType.BINARY_EXPR and condition.op == "<":
                # 获取左右操作数
                left = condition.left  # 通常是变量 i
                right = condition.right  # 通常是上限 n
                
                # 创建range调用
                range_id = node_factory.create_identifier("range")
                range_args = node_factory.create_expression_list([right])
                range_call = node_factory.create_call_expr(range_id, range_args)
                
                # 创建for-in语句
                for_stmt = node_factory.create_for_in_stmt(
                    node_factory.create_declarator_type(node_factory.create_type_identifier("let")),
                    left,  # 迭代变量
                    range_call,  # 可迭代对象
                    body,  # 循环体
                    "in",  # Python使用in关键字
                    False  # 非异步
                )
                new_stmts.append(for_stmt)
            else:
                # 如果条件不是简单的 i < n 形式，则保持原样
                for_stmt = node_factory.create_for_stmt(body, condition=condition)
                new_stmts.append(for_stmt)
        else:
            # 原始C/C++/Java/JS风格的转换
            for_stmt = node_factory.create_for_stmt(body, condition=condition)
            new_stmts.append(for_stmt)

        return (True, new_stmts)
