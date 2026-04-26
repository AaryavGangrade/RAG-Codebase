import ast
import os
from typing import List, Dict

class CodebaseParser:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _extract_imports(self, node: ast.AST) -> List[str]:
        imports = []
        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    imports.append(alias.name)
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    imports.append(n.module)
        return imports

    def parse_file(self, filepath: str) -> List[Dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        imports = self._extract_imports(tree)
        extracted = []

        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            extracted.append({
                "type": "module",
                "name": os.path.basename(filepath),
                "file": filepath,
                "code": source,
                "docstring": module_docstring,
                "imports": imports,
                "start_line": 1,
                "end_line": len(source.splitlines())
            })

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                docstring = ast.get_docstring(node)
                func_code = ast.get_source_segment(source, node)
                if not func_code:
                    continue
                extracted.append({
                    "type": "function",
                    "name": node.name,
                    "file": filepath,
                    "code": func_code,
                    "docstring": docstring or "",
                    "imports": imports,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno
                })
            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                class_code = ast.get_source_segment(source, node)
                if not class_code:
                    continue
                extracted.append({
                    "type": "class",
                    "name": node.name,
                    "file": filepath,
                    "code": class_code,
                    "docstring": docstring or "",
                    "imports": imports,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno
                })

        return extracted

    def parse_codebase(self) -> List[Dict]:
        extracted = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    extracted.extend(self.parse_file(filepath))
        return extracted
