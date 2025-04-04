'''
FileName: 
Description: 
Autor: Liujunjie/Aries-441
Github: https://github.com/Aries-441
Date: 2025-04-04 17:52:02
E-mail: sjtu.liu.jj@gmail.com/sjtu.1518228705@sjtu.edu.cn
LastEditTime: 2025-04-04 17:56:11
'''
from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import (  
    BinopUpdateVisitor,
    AssignUpdateVisitor,
)


class UpdateTransformer(CodeTransformer):
    name = "UpdateTransformer"
    TRANSFORM_BINOP_UPDATE = "UpdateTransformer.binop_update"
    TRANSFORM_ASSIGN_UPDATE = "UpdateTransformer.assign_update"

    def __init__(self, lang: str = "c") -> None:
        super().__init__()
        self.lang = lang

    def get_available_transforms(self):
        return [
            self.TRANSFORM_BINOP_UPDATE,
            self.TRANSFORM_ASSIGN_UPDATE,
        ]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_BINOP_UPDATE: BinopUpdateVisitor(),
            self.TRANSFORM_ASSIGN_UPDATE: AssignUpdateVisitor(),
        }[dst_style].visit(node)
