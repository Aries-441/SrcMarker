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
    # 示例 1：基本字面量匹配
    value = 2
    match value:
        case 1:
            print("Value is 1")
        case 2:
            print("Value is 2")
        case _:
            print("Value is something else")

    # 示例 2：变量匹配
    name = "Alice"
    match name:
        case "Bob":
            print("Hello Bob!")
        case "Alice":
            print("Hello Alice!")
        case _:
            print("Hello stranger!")

    # 示例 3：序列解构
    point = (1, 2)
    match point:
        case (0, 0):
            print("Origin")
        case (x, 0):
            print(f"On the x-axis at x={x}")
        case (0, y):
            print(f"On the y-axis at y={y}")
        case (x, y):
            print(f"Point is at ({x}, {y})")
        case _:
            print("Not a point")

    # 示例 4：类匹配
    class Point:
        x: int
        y: int

    p = Point()
    p.x = 1
    p.y = 2

    match p:
        case Point(x=0, y=0):
            print("Origin")
        case Point(x=x_val, y=0):
            print(f"On the x-axis at x={x_val}")
        case Point(x=0, y=y_val):
            print(f"On the y-axis at y={y_val}")
        case Point():
            print("Point is somewhere else")
        case _:
            print("Not a Point")

    # 示例 5：组合匹配
    shape = ("circle", 5)
    match shape:
        case ("circle", radius):
            print(f"Circle with radius {radius}")
        case ("rectangle", width, height):
            print(f"Rectangle with width {width} and height {height}")
        case ("square", size) if size > 0:
            print(f"Square with size {size}")
        case _:
            print("Unknown shape")

    # 示例 6：匹配剩余元素
    numbers = [1, 2, 3, 4, 5]
    match numbers:
        case [first, second, *rest]:
            print(f"First: {first}, Second: {second}, Rest: {rest}")
        case _:
            print("Not a list with at least two elements")

'''


    code1 = '''
    # 使用 for 循环
text = "hello"
for char in text:
    print(char)
'''

    code2 = '''
# 1. 不带参数的重新抛出异常
raise

# 2. 抛出指定异常
raise ValueError

# 3. 带消息的异常
raise Exception("An error occurred")

# 4. 带原因的异常
raise RuntimeError("Failed to process data") from ValueError("Invalid input")
            '''
            
    code3 = '''
# 1. 简单的平方计算
square = lambda x: x ** 2

# 2. 条件表达式
is_even = lambda x: "even" if x % 2 == 0 else "odd"

# 3. 嵌套 lambda
nested_lambda = lambda x: lambda y: x + y

# 4. 使用 lambda 进行映射操作
square_list = list(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))

# 5. 使用 lambda 进行排序
sorted_list = sorted([3, 1, 4, 1, 5, 9, 2, 6], key=lambda x: -x)

# 6. 使用 lambda 捕获异常（间接方式）
safe_divide = lambda a, b: (lambda: a / b)() if b != 0 else "Division by zero"
            '''
            
    code4 = '''
    op: NodeOp = lisp_ast.op
    '''
    
    code5 = '''
            try:
                val = int(val)
            except ValueError:
            '''
    
    code6 = '''
                tables_list_resp = self.service.tables().list(projectId=project_id)
                tables_list_resp = self.service.tables().\
                list(projectId=project_id)

    '''
    
    code7= '''

    '''
    
    # 测试语法树
    tree = parser.parse(bytes(code7, 'utf8'))
    print_tree(tree.root_node)  
