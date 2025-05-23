from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import ForToWhileVisitor, WhileToForVisitor


class LoopTransformer(CodeTransformer):
    name = "LoopTransformer"
    TRANSFORM_LOOP_FOR = "LoopTransformer.for_loop"
    TRANSFORM_LOOP_WHILE = "LoopTransformer.while_loop"

    def __init__(self, lang: str = "c") -> None:
        super().__init__()
        self.lang = lang

    def get_available_transforms(self):
        return [self.TRANSFORM_LOOP_FOR, self.TRANSFORM_LOOP_WHILE]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_LOOP_FOR: WhileToForVisitor(self.lang),
            self.TRANSFORM_LOOP_WHILE: ForToWhileVisitor(self.lang),
        }[dst_style].visit(node)
