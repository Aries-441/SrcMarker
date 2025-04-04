'''
FileName: 
Description: 
Autor: Liujunjie/Aries-441
Github: https://github.com/Aries-441
Date: 2025-03-12 11:49:57
E-mail: sjtu.liu.jj@gmail.com/sjtu.1518228705@sjtu.edu.cn
LastEditTime: 2025-03-25 07:23:05
'''
from tree_sitter import Parser, Node, Language
import os

# 初始化解析器
if os.name == 'nt':  # Windows 系统
    language_lib_file = "C:\\Users\\15182\\Desktop\\tree-sitter\\languages.dll"  # 根据实际路径修改
elif os.name == 'posix':  # Linux 或 macOS 系统
    language_lib_file = "/mnt/e/SrcMarker/parser/languages.so"  
else:
    raise OSError("Unsupported operating system")

JS_LANGUAGE = Language(language_lib_file, 'python')
parser = Parser()
parser.set_language(JS_LANGUAGE)


# 使用示例
if __name__ == '__main__':
    
    def print_tree(node, indent=0, field_name=""):
        # 打印当前节点的类型和字段名（如果有）
        field_info = f" [{field_name}]" if field_name else ""
        print("  " * indent + f"{node.type}{field_info}")

        # 遍历子节点，并获取其字段名
        for i, child in enumerate(node.children):
            # 通过child的索引获取对应的字段名
            childField = node.field_name_for_child(i)
            print_tree(child, indent + 1, childField) 

    code = '''
not a
'''
    
    # 测试语法树
    tree = parser.parse(bytes(code, 'utf8'))
    print_tree(tree.root_node)  
    