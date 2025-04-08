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
from mutable_tree.adaptors.lang.python_adaptor import unhandled_types_counter, unhandled_instances, record_unhandled_type

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


def compare_tokens(old_tokens: List[str], new_tokens: List[str]) -> bool:
    """比较两个token列表是否语义等价
    
    参数:
        old_tokens: 原始token列表
        new_tokens: 新生成的token列表
        
    返回:
        bool: 如果token列表语义等价则返回True，否则返回False
    """
    def preprocess_tokens(tokens: List[str]) -> List[str]:
        """预处理token列表"""
        processed = []
        i = 0
        while i < len(tokens):
            # 处理续行符
            if tokens[i] == '\\\n' and i + 1 < len(tokens):
                i += 2  # 跳过续行符和后面的换行符
                continue
            
            # 处理链式调用中的换行
            if (i + 2 < len(tokens) and 
                tokens[i] == ')' and 
                tokens[i+1].strip() == '' and  # 换行
                tokens[i+2] == '.'):  # 下一行以点号开始
                processed.append(tokens[i])  # 保留右括号
                i += 2  # 跳过换行，保留下一个点号
                continue
            
            # 处理字符串引号差异
            if tokens[i].startswith(('"', "'")):
                # 统一转换为单引号比较内容
                content = tokens[i][1:-1]
                processed.append(f"'{content}'")
            else:
                # 处理赋值操作符周围的空格
                if '=' in tokens[i]:
                    processed.append(tokens[i].replace(' ', ''))
                else:
                    processed.append(tokens[i])
            i += 1
        return processed
    
    # 预处理token列表
    old_processed = preprocess_tokens(old_tokens)
    new_processed = preprocess_tokens(new_tokens)
    
    # 比较处理后的token列表
    if len(old_processed) != len(new_processed):
        diff = abs(len(old_processed) - len(new_processed))
        if diff <= 3:
            logger = logging.getLogger("dataset_filter_python")
            logger.debug("新旧Tokens差异:")
            logger.debug(f"原始Tokens: {old_processed}")
            logger.debug(f"复原Tokens: {new_processed}")
        
        # 尝试对齐逗号
        i, j = 0, 0
        while i < len(old_processed) and j < len(new_processed):
            if old_processed[i] == new_processed[j]:
                i += 1
                j += 1
            elif old_processed[i] == ',':
                # 检查是否在参数列表、数组或字典的末尾
                if i + 1 < len(old_processed) and old_processed[i+1] in [')', ']', '}']:
                    i += 1  # 跳过多余的逗号
                else:
                    return False
            elif new_processed[j] == ',': 
                # 检查是否在参数列表、数组或字典的末尾
                if j + 1 < len(new_processed) and new_processed[j+1] in [')', ']', '}']:
                    j += 1  # 跳过多余的逗号
                else:
                    return False
            else:
                return False
        
        # 检查剩余token是否都是逗号
        remaining_old = all(t == ',' for t in old_processed[i:])
        remaining_new = all(t == ',' for t in new_processed[j:])
        return remaining_old and remaining_new
    
    # 长度相同则逐个比较
    mismatch_count = 0
    mismatch_indices = []
    for idx, (old, new) in enumerate(zip(old_processed, new_processed)):
        if old != new:
            mismatch_count += 1
            mismatch_indices.append(idx)
            # 忽略字符串引号差异
            if ((old.startswith('"') and old.endswith('"') and 
                 new.startswith("'") and new.endswith("'")) or
                (old.startswith("'") and old.endswith("'") and 
                 new.startswith('"') and new.endswith('"'))):
                mismatch_count -= 1  # 不计入实际差异
                continue
            return False
    
    if abs(mismatch_count) <= 3 and abs(mismatch_count) > 0:
        logger = logging.getLogger("dataset_filter_python")
        logger.debug("新旧Tokens差异:")
        for idx in mismatch_indices:
            # 只打印差异的token及其上下文
            start = max(0, idx - 2)
            end = min(len(old_processed), idx + 3)
            logger.debug(f"位置 {idx}:")
            logger.debug(f"原始: {' '.join(old_processed[start:end])}")
            logger.debug(f"复原:   {' '.join(new_processed[start:end])}")
            logger.debug("-" * 50)
    
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
        
        # 用于统计未处理的类型
        unhandled_types = {}
        
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
                # 在处理每个实例前，记录当前实例ID
                if 'current_instance_id' in function_round_trip.__dict__:
                    current_id = function_round_trip.current_instance_id
                else:
                    current_id = None
                
                # 修改record_unhandled_type函数的调用，传入实例ID
                record_unhandled_type.instance_id = current_id
                
                mutable_root = PythonAdaptor.convert_function_definition(func_root)
                # 添加AST打印功能，但使用更简洁的格式
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("生成的AST结构:")
                    # 限制AST结构的输出深度，避免日志过大
                    ast_str = print_ast_structure(mutable_root, max_depth=10)
                    # 将AST结构分行记录，避免一行过长
                    for line in ast_str.split('\n'):
                        if line.strip():
                            logger.debug(line)
            except Exception as e:
                error_msg = f"Python转换错误: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                # 记录原始代码，便于调试
                logger.error(f"原始代码:\n{code}")
                
                # 捕获未处理的类型错误
                error_str = str(e)
                # 检查是否包含未处理的语句类型或表达式类型
                for error_pattern in ["未处理的语句类型", "未处理的表达式类型"]:
                    if error_pattern in error_str:
                        # 提取未处理的类型
                        import re
                        match = re.search(f"{error_pattern}: (\\w+)", error_str)
                        if match:
                            type_name = match.group(1)
                            error_key = f"{error_pattern}: {type_name}"
                            
                            # 更新统计信息
                            if error_key in unhandled_types:
                                unhandled_types[error_key] += 1
                            else:
                                unhandled_types[error_key] = 1
                            
                            # 将统计信息传递给全局计数器
                            if not hasattr(function_round_trip, 'unhandled_types_counter'):
                                function_round_trip.unhandled_types_counter = Counter()
                            
                            function_round_trip.unhandled_types_counter[error_key] += 1
                
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
            logger.error(f"原始代码:\n{code}")
            return (False, "new code has error")

        old_tokens = collect_tokens(tree.root_node)
        new_tokens = collect_tokens(new_root)
        
        # 简化Token数量记录
        logger.debug(f"Token数量: 原始={len(old_tokens)}, 生成={len(new_tokens)}, 差异={len(new_tokens) - len(old_tokens)}")
        
        # 记录原始代码和生成的代码，但不记录详细的Token比较
        logger.debug(f"原始代码:\n{code}")
        logger.debug(f"生成的代码:\n{new_code}")
        
        if not compare_tokens(old_tokens, new_tokens) and lang != "javascript":
            # 简化不匹配信息的记录
            logger.error("Token不匹配")
            
            # 只记录Token数量差异，不记录详细的Token比较
            if len(old_tokens) != len(new_tokens):
                logger.error(f"Token数量不同: 原始={len(old_tokens)}, 生成={len(new_tokens)}")
            else:
                # 只记录不匹配的Token数量，不记录详细内容
                mismatch_count = sum(1 for old, new in zip(old_tokens, new_tokens) if old != new)
                logger.error(f"有 {mismatch_count} 个Token不匹配")
            
            return (False, "token mismatch")

        return (True, None)
    except Exception as e:
        # 捕获所有未处理的异常
        error_msg = f"未预期的错误: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return (False, f"unexpected error: {str(e)}")

# 初始化实例ID
function_round_trip.current_instance_id = None

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

            # 设置当前实例ID
            function_round_trip.current_instance_id = idx
            
            # 简化每个实例的日志记录
            logger.debug(f"处理实例 #{idx+1}")
            
            success, msg = function_round_trip(parser, code, lang, logger)
            
            if not success:
                tot_fail += 1
                errors.append(msg)
                
                # 简化错误记录
                logger.warning(f"实例 #{idx} 处理失败: {msg}")
                
                # 记录每种错误类型的前几个示例
                if msg not in error_examples:
                    error_examples[msg] = []
                    logger.info(f"发现新错误类型: {msg}")
                
                if len(error_examples[msg]) < 3:  # 每种错误类型最多记录3个示例
                    error_examples[msg].append((idx, code))
                    # 使用分隔符使示例更清晰
                    logger.info(f"错误类型 '{msg}' 的示例 #{idx}:\n{code}\n{'='*50}")
            else:
                data_instance["original_string"] = code
                fo.write(json.dumps(data_instance) + "\n")
                tot_good += 1
                
                # 减少成功处理的日志记录频率
                if tot_good % 500 == 0:
                    logger.info(f"已成功处理 {tot_good} 个实例")
            
            tot_funcs += 1
            
            # 减少统计信息的记录频率
            if tot_funcs % 5000 == 0:
                logger.info(f"当前统计 - 成功: {tot_good}, 失败: {tot_fail}, 总计: {tot_funcs}")
                # 只记录主要错误类型
                temp_counter = Counter(errors)
                top_errors = temp_counter.most_common(5)
                for err_msg, count in top_errors:
                    logger.info(f"主要错误类型: {err_msg}, 当前数量: {count}")

        # 处理完成后记录最终统计信息
        msg_counter = Counter(errors)
        print(f"Good: {tot_good}, Fail: {tot_fail}, Total: {tot_funcs}")
        for msg, count in msg_counter.most_common():
            print(f"{msg}: {count}")
            
        # 添加未处理类型的汇总报告
        if lang == "python" and unhandled_types_counter:
            print("\n未处理类型汇总报告:")
            print(f"共有 {len(unhandled_types_counter)} 种未处理的类型")
            
            for type_error, count in unhandled_types_counter.most_common():
                instances_count = len(unhandled_instances.get(type_error, set()))
                print(f"  {type_error}: 出现 {count} 次，影响 {instances_count} 个实例")
            
            # 同时记录到日志
            logger.info("\n未处理类型汇总报告:")
            for type_error, count in unhandled_types_counter.most_common():
                instances_count = len(unhandled_instances.get(type_error, set()))
                logger.info(f"  {type_error}: 出现 {count} 次，影响 {instances_count} 个实例")
    
    print(f"Output written to {output_file_path}")
    print(f"Error log written to {lang}_filter_errors.log")


if __name__ == "__main__":
    main(sys.argv[1:])