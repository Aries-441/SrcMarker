import os
import re
import sys
import json
import tree_sitter
import traceback
import logging
from tqdm import tqdm
from collections import Counter
from mutable_tree.adaptors import JavaAdaptor, CppAdaptor, JavaScriptAdaptor, PythonAdaptor
from mutable_tree.stringifiers import (
    JavaStringifier,
    CppStringifier,
    JavaScriptStringifier,
    PythonStringifier,
)
from typing import List, Tuple
from mutable_tree.nodes import Node

# 添加在文件中的适当位置
# 添加在文件中的适当位置
def print_ast_structure(node, indent=0, max_depth=20):
    """递归打印AST结构"""
    if indent > max_depth:
        return "  " * indent + "... (达到最大深度)\n"
    
    if node is None:
        return "  " * indent + "None\n"
    
    result = "  " * indent + f"{node.__class__.__name__}"
    
    # 添加节点类型和其他关键属性
    if hasattr(node, 'node_type'):
        result += f" (type={node.node_type})"
    
    # 添加文档字符串
    if hasattr(node, 'docstring') and node.docstring:
        result += f" docstring='{node.docstring[:30]}...'" if len(node.docstring) > 30 else f" docstring='{node.docstring}'"
    
    result += "\n"
    
    # 递归处理子节点
    try:
        if hasattr(node, 'get_children') and callable(node.get_children):
            children = node.get_children()
            children_names = node.get_children_names() if hasattr(node, 'get_children_names') and callable(node.get_children_names) else [""] * len(children)
            
            for i, (child, name) in enumerate(zip(children, children_names)):
                result += "  " * (indent + 1) + f"{name}: " if name else ""
                result += print_ast_structure(child, indent + 1, max_depth)
    except NotImplementedError:
        # 处理未实现 get_children 方法的节点
        result += "  " * (indent + 1) + f"<未实现 get_children 方法>\n"
    
    # 处理特殊属性
    special_attrs = ['header', 'body', 'stmts', 'statements', 'expression', 'declarator', 'parameters', 'docstring']
    for attr in special_attrs:
        if hasattr(node, attr) and getattr(node, attr) is not None:
            # 检查属性是否已经在 children 中处理过
            try:
                if hasattr(node, 'get_children_names') and callable(node.get_children_names) and attr in node.get_children_names():
                    continue
            except NotImplementedError:
                pass  # 如果 get_children_names 未实现，继续处理
            
            # 处理特殊属性
            attr_value = getattr(node, attr)
            result += "  " * (indent + 1) + f"{attr}: {print_ast_structure(attr_value, indent + 1, max_depth)}"
    
    # 添加对简单属性的处理
    simple_attrs = ['name', 'value', 'operator', 'type']
    for attr in simple_attrs:
        if hasattr(node, attr) and getattr(node, attr) is not None:
            attr_value = getattr(node, attr)
            if not isinstance(attr_value, Node):  # 只处理非节点类型的简单属性
                result += "  " * (indent + 1) + f"{attr}: {attr_value}\n"
    
    return result

# 配置日志记录
def setup_logging(lang: str) -> logging.Logger:
    logger = logging.getLogger(f"dataset_filter_{lang}")
    logger.setLevel(logging.DEBUG)
    
    # 直接使用项目根目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件处理器 - 直接保存在项目根目录
    log_file = os.path.join(project_dir, f"{lang}_filter_errors.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    
    return logger


def collect_tokens(root: tree_sitter.Node) -> List[str]:
    tokens: List[str] = []

    def _collect_tokens(node: tree_sitter.Node):
        if node.child_count == 0:
            tokens.append(node.text.decode())

        for ch in node.children:
            _collect_tokens(ch)

    _collect_tokens(root)
    return tokens


def remove_comments(source: str):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/") or s.startswith("#"):
            return " "  # 注意：返回空格而不是空字符串
        elif s.startswith('"""') or s.startswith("'''"):
            # 处理Python的文档字符串
            return " "
        else:
            return s

    # 修改正则表达式以包含Python的文档字符串
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|#.*?$|\'\'\'[\s\S]*?\'\'\'|"""[\s\S]*?"""|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split("\n"):
        if x.strip() != "":
            temp.append(x)
    return "\n".join(temp)

def _get_java_function_root(root: tree_sitter.Node) -> tree_sitter.Node:
    assert root.type == "program"
    class_decl_node = root.children[0]
    assert class_decl_node.type == "class_declaration"
    class_body_node = class_decl_node.children[3]
    assert class_body_node.type == "class_body"
    func_root_node = class_body_node.children[1]
    assert func_root_node.type == "method_declaration", func_root_node.type
    return func_root_node


def _get_cpp_function_root(root: tree_sitter.Node) -> tree_sitter.Node:
    assert root.type == "translation_unit"
    func_root_node = root.children[0]
    assert func_root_node.type == "function_definition"
    return func_root_node


def _get_javascript_function_root(root: tree_sitter.Node) -> tree_sitter.Node:
    assert root.type == "program"
    func_root_node = root.children[0]

    if func_root_node.type == "function_declaration":
        return func_root_node
    elif func_root_node.type == "expression_statement":
        func_root_node = func_root_node.children[0]
        assert func_root_node.type == "function", func_root_node.type
        return func_root_node
    elif func_root_node.type == "generator_function_declaration":
        return func_root_node
    else:
        raise RuntimeError(f"Unexpected root node type: {func_root_node.type}")


def _get_python_function_root(root: tree_sitter.Node) -> tree_sitter.Node:
    assert root.type == "module"
    for child in root.children:
        if child.type == "function_definition":
            return child
    raise RuntimeError("No function definition found in Python module")


def compare_tokens(old_tokens: List[str], new_tokens: List[str]):
    if len(old_tokens) != len(new_tokens):
        return False
    for i, (old_token, new_token) in enumerate(zip(old_tokens, new_tokens)):
        if old_token != new_token:
            return False
    return True


def function_round_trip(parser: tree_sitter.Parser, code: str, lang: str, logger: logging.Logger) -> Tuple[bool, str]:
    if lang == "java":
        wrapped = f"public class A {{ {code} }}"
    else:
        wrapped = code

    try:
        tree = parser.parse(wrapped.encode("utf-8"))
        if tree.root_node.has_error:
            logger.error(f"原始代码解析错误: {tree.root_node}")
            return (False, "original code has error")

        assert not tree.root_node.has_error
        
        if lang == "java":
            func_root = _get_java_function_root(tree.root_node)
            try:
                mutable_root = JavaAdaptor.convert_function_declaration(func_root)
            except Exception as e:
                error_msg = f"Java转换错误: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                return (False, str(e))
            stringifier = JavaStringifier()
        elif lang == "cpp":
            func_root = _get_cpp_function_root(tree.root_node)
            try:
                mutable_root = CppAdaptor.convert_function_definition(func_root)
            except Exception as e:
                error_msg = f"C++转换错误: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                return (False, str(e))
            stringifier = CppStringifier()
        elif lang == "javascript":
            func_root = _get_javascript_function_root(tree.root_node)
            try:
                mutable_root = JavaScriptAdaptor.convert_function_declaration(func_root)
            except Exception as e:
                error_msg = f"JavaScript转换错误: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                return (False, str(e))
            stringifier = JavaScriptStringifier()
        elif lang == "python":
            func_root = _get_python_function_root(tree.root_node)
            try:
                mutable_root = PythonAdaptor.convert_function_definition(func_root)
                # 添加AST打印功能
                logger.debug("生成的AST结构:")
                logger.debug(print_ast_structure(mutable_root))
            except Exception as e:
                error_msg = f"Python转换错误: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                # 记录原始代码，便于调试
                logger.error(f"原始代码:\n{code}")
                return (False, str(e))
            # 使用专门的 PythonStringifier 替代 JavaScriptStringifier
            stringifier = PythonStringifier()
        else:
            raise ValueError(f"不支持的语言: {lang}")

        new_code = stringifier.stringify(mutable_root)
        if lang == "java":
            new_code = f"public class A {{ {new_code} }}"

        new_root = parser.parse(new_code.encode("utf-8")).root_node
        if new_root.has_error:
            logger.error(f"生成的代码解析错误: {new_root}")
            logger.error(f"生成的代码:\n{new_code}")
            return (False, "new code has error")

        old_tokens = collect_tokens(tree.root_node)
        new_tokens = collect_tokens(new_root)
        
        # 记录变换前后的Token数量
        logger.debug(f"原始代码Token数量: {len(old_tokens)}")
        logger.debug(f"生成代码Token数量: {len(new_tokens)}")
        
        # 如果Token数量不同，直接记录新生成的代码
        if len(old_tokens) != len(new_tokens):
            logger.debug(f"Token数量差异: {len(new_tokens) - len(old_tokens)}")
            logger.debug(f"原始代码:\n{code}")
            logger.debug(f"生成的代码:\n{new_code}")
        else:
            # 只有在Token数量相同时才比较前几个Token
            max_tokens_to_log = min(10, len(old_tokens))
            for i in range(max_tokens_to_log):
                logger.debug(f"Token {i}: 原始='{old_tokens[i]}', 新生成='{new_tokens[i]}'")
        
        if not compare_tokens(old_tokens, new_tokens) and lang != "javascript":
            # NOTE: we ignore token mismatches for js
            # because semicolons in js are optional and they cause tons of mismatches
            # NOTE: we ignore token mismatches for python
            # because python uses indentation to denote code blocks
            
            # 使用ERROR级别记录不匹配信息，但不重复记录已经在DEBUG级别记录的详细信息
            logger.error("Token不匹配:")
            if len(old_tokens) != len(new_tokens):
                logger.error(f"Token数量不同: 原始={len(old_tokens)}, 新生成={len(new_tokens)}")
                # 不再重复记录生成的代码，因为已经在DEBUG级别记录过了
            else:
                # 记录不匹配的Token
                mismatch_count = 0
                for i, (old_token, new_token) in enumerate(zip(old_tokens, new_tokens)):
                    if old_token != new_token:
                        logger.error(f"Token {i} 不匹配: 原始='{old_token}', 新生成='{new_token}'")
                        mismatch_count += 1
                        # 只记录前10个不匹配的Token，避免日志过大
                        if mismatch_count >= 10:
                            logger.error(f"还有更多不匹配的Token未显示...")
                            break
            return (False, "token mismatch")

        return (True, None)
    except Exception as e:
        # 捕获所有未处理的异常
        error_msg = f"未预期的错误: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return (False, f"unexpected error: {str(e)}")


def main(args):
    if len(args) != 2:
        print("Usage: python dataset_filter.py <lang> <file>")
        return
    lang, file_path = args

    # 设置日志记录
    logger = setup_logging(lang)
    logger.info(f"开始处理 {lang} 文件: {file_path}")

    parser = tree_sitter.Parser()
    parser.set_language(tree_sitter.Language("./parser/languages.so", lang))

    errors = []
    tot_good = 0
    tot_fail = 0
    tot_funcs = 0
    error_examples = {}  # 记录每种错误类型的示例

    with open(file_path, "r", encoding="utf-8") as fi:
        lines = fi.readlines()
        data_instances = [json.loads(line) for line in lines]
    
    logger.info(f"共加载 {len(data_instances)} 个数据实例")

    filename_noext = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(
        os.path.dirname(file_path), f"{filename_noext}_filtered.jsonl"
    )
    
    with open(output_file_path, "w", encoding="utf-8") as fo:
        for idx, data_instance in enumerate(tqdm(data_instances)):
            code = data_instance["original_string"]
            code = remove_comments(code)

            # 每处理100个实例记录一次进度
            if idx % 100 == 0:
                logger.info(f"正在处理: {idx}/{len(data_instances)} 实例")
            
            # 对每个实例进行更详细的日志记录
            logger.debug(f"处理第 {idx+1} 个实例")
            
            success, msg = function_round_trip(parser, code, lang, logger)
            
            if not success:
                tot_fail += 1
                errors.append(msg)
                
                # 实时记录错误信息
                logger.warning(f"实例 #{idx} 处理失败: {msg}")
                
                # 记录每种错误类型的前几个示例
                if msg not in error_examples:
                    error_examples[msg] = []
                    # 第一次遇到这种错误类型时记录
                    logger.info(f"发现新错误类型: {msg}")
                
                if len(error_examples[msg]) < 3:  # 每种错误类型最多记录3个示例
                    error_examples[msg].append((idx, code))
                    # 记录错误示例
                    logger.info(f"错误类型 '{msg}' 的示例 #{idx}:\n{code}\n{'='*50}")
            else:
                data_instance["original_string"] = code
                fo.write(json.dumps(data_instance) + "\n")
                tot_good += 1
                
                # 每处理成功100个实例记录一次
                if tot_good % 100 == 0:
                    logger.info(f"已成功处理 {tot_good} 个实例")
            
            tot_funcs += 1
            
            # 每处理1000个实例记录一次统计信息
            if tot_funcs % 1000 == 0:
                logger.info(f"当前统计 - 成功: {tot_good}, 失败: {tot_fail}, 总计: {tot_funcs}")
                # 记录当前错误类型分布
                temp_counter = Counter(errors)
                for err_msg, count in temp_counter.items():
                    logger.info(f"错误类型: {err_msg}, 当前数量: {count}")

        # 处理完成后记录最终统计信息
        msg_counter = Counter(errors)
        print(f"Good: {tot_good}, Fail: {tot_fail}, Total: {tot_funcs}")
        for msg, count in msg_counter.items():
            print(f"{msg}: {count}")
            
        # 记录最终错误统计
        logger.info(f"处理完成 - 成功: {tot_good}, 失败: {tot_fail}, 总计: {tot_funcs}")
        for msg, count in msg_counter.items():
            logger.info(f"最终错误类型统计: {msg}, 数量: {count}")
    
    print(f"Output written to {output_file_path}")
    print(f"Error log written to {lang}_filter_errors.log")  # 修改这一行，移除 logs/ 前缀


if __name__ == "__main__":
    main(sys.argv[1:])
