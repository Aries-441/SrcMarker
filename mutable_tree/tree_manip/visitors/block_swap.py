from .visitor import TransformingVisitor
from mutable_tree.nodes import Node, NodeType, node_factory
from mutable_tree.nodes import IfStatement
from mutable_tree.nodes import (
    Expression,
    UnaryExpression,
    BinaryExpression,
    ParenthesizedExpression,
    AssignmentExpression,
)
from mutable_tree.nodes import BinaryOps, UnaryOps
from typing import Optional


class BlockSwapper(TransformingVisitor):
    def _can_swap_condition(self, expr: Expression):
        raise NotImplementedError()

    def _condition_transform(self, expr: Expression):
        raise NotImplementedError()

    def visit_IfStatement(
        self,
        node: IfStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)

        condition = node.condition
        consequence = node.consequence
        alternate = node.alternate

        if alternate is None:
            return (False, [])

        if not self._can_swap_condition(condition):
            return (False, [])

        new_condition = self._condition_transform(condition)

        if alternate.node_type != NodeType.BLOCK_STMT:
            alternate = node_factory.wrap_block_stmt(alternate)
        if consequence.node_type != NodeType.BLOCK_STMT:
            consequence = node_factory.wrap_block_stmt(consequence)

        new_node = node_factory.create_if_stmt(new_condition, alternate, consequence)
        return (True, [new_node])


class NormalBlockSwapper(BlockSwapper):
    # prefers le/lt; prefers non-negated conditions
    def _can_swap_condition(self, expr: Expression):
        while isinstance(expr, ParenthesizedExpression):
            expr = expr.expr

        if isinstance(expr, BinaryExpression):
            op = expr.op
            if op in {BinaryOps.GE, BinaryOps.GT, BinaryOps.NE,
                      BinaryOps.NOTIN, BinaryOps.ISNOT,}:
                return True

        elif isinstance(expr, UnaryExpression):
            op = expr.op
            if op in [UnaryOps.NOT, UnaryOps.PYNOT]:
                return True
            
        # Python 特有：处理 if x 这种直接判断真值的情况
        elif expr.node_type in {
            NodeType.CALL_EXPR,
            NodeType.IDENTIFIER,
            NodeType.LITERAL
        }:
            return True

        return False

    def _condition_transform(self, expr: Expression):
        while isinstance(expr, ParenthesizedExpression):
            expr = expr.expr

        if isinstance(expr, BinaryExpression):
            op = expr.op
            new_op = {
                # take complement
                BinaryOps.GE: BinaryOps.LT,
                BinaryOps.GT: BinaryOps.LE,
                BinaryOps.NE: BinaryOps.EQ,
                BinaryOps.ISNOT: BinaryOps.IS,
                BinaryOps.NOTIN: BinaryOps.IN,
            }[op]
            return node_factory.create_binary_expr(expr.left, expr.right, new_op)

        elif isinstance(expr, UnaryExpression):
            assert expr.op in [UnaryOps.NOT, UnaryOps.PYNOT]
            return expr.operand
        
        # Python 特有：处理直接真值判断的情况
        elif expr.node_type in {NodeType.CALL_EXPR, NodeType.IDENTIFIER, NodeType.LITERAL}:
            return node_factory.create_unary_expr(expr, UnaryOps.PYNOT)

        else:
            raise RuntimeError(f"cannot transform expression {expr}")


class NegatedBlockSwapper(BlockSwapper):
    # prefers ge/gt; prefers negated conditions
    def _can_swap_condition(self, expr: Expression):
        while isinstance(expr, ParenthesizedExpression):
            expr = expr.expr

        if isinstance(expr, BinaryExpression):
            op = expr.op
            if op in {BinaryOps.LE, BinaryOps.LT, BinaryOps.EQ,
                      BinaryOps.IS, BinaryOps.IN}:
                return True
            else:
                return False

        elif isinstance(expr, UnaryExpression):
            op = expr.op
            if op in [UnaryOps.NOT, UnaryOps.PYNOT]:
                # dont negate twice
                return False

        elif isinstance(expr, AssignmentExpression):
            return False

        return True

    def _condition_transform(self, expr: Expression):
        while isinstance(expr, ParenthesizedExpression):
            expr = expr.expr

        if isinstance(expr, BinaryExpression):
            op = expr.op
            new_op = {
                BinaryOps.LE: BinaryOps.GT,
                BinaryOps.LT: BinaryOps.GE,
                BinaryOps.EQ: BinaryOps.NE,
                BinaryOps.IS: BinaryOps.ISNOT,
                BinaryOps.IN: BinaryOps.NOTIN,
            }[op]
            return node_factory.create_binary_expr(expr.left, expr.right, new_op)

        else:
            if expr.node_type not in {
                NodeType.CALL_EXPR,
                NodeType.IDENTIFIER,
                NodeType.LITERAL,
                NodeType.UNARY_EXPR,
            }:
                expr = node_factory.create_parenthesized_expr(expr)
            return node_factory.create_unary_expr(expr, UnaryOps.NOT)
