#!/usr/bin/env python3
"""
NumPy Test Porting Script

Legacy scaffolding only: generated Rust stubs do not establish compatibility.
Review source provenance and assertions as described in docs/TEST_PORTING.md.

This script helps port NumPy tests to Raptors by:
1. Parsing NumPy test files
2. Identifying test functions
3. Generating Rust test stubs
4. Mapping NumPy API calls to Raptors API
5. Tracking porting progress
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class NumPyTestParser:
    """Parse NumPy test files and extract test information"""
    
    def __init__(self, test_file: Path):
        self.test_file = test_file
        self.test_functions: List[Dict] = []
        self.imports: List[str] = []
        
    def parse(self) -> None:
        """Parse the test file and extract test functions"""
        with open(self.test_file, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            self._extract_imports(tree)
            self._extract_tests(tree)
        except SyntaxError as e:
            print(f"Error parsing {self.test_file}: {e}")
            return
    
    def _extract_imports(self, tree: ast.AST) -> None:
        """Extract import statements"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    self.imports.append(f"{module}.{alias.name}")
    
    def _extract_tests(self, tree: ast.AST) -> None:
        """Extract test functions and classes"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    self.test_functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node),
                    })
            elif isinstance(node, ast.ClassDef):
                if node.name.startswith('Test'):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                            self.test_functions.append({
                                'name': f"{node.name}::{item.name}",
                                'line': item.lineno,
                                'docstring': ast.get_docstring(item),
                            })


class RustTestGenerator:
    """Generate Rust test code from NumPy test information"""
    
    def __init__(self):
        self.api_mapping = {
            'np.array': 'Array::new',
            'np.zeros': 'zeros',
            'np.ones': 'ones',
            'np.empty': 'empty',
            'np.testing.assert_array_equal': 'assert_array_equal',
            'np.testing.assert_allclose': 'assert_allclose',
        }
    
    def generate_test_stub(self, test_info: Dict) -> str:
        """Generate a Rust test stub from test information"""
        test_name = test_info['name'].replace('::', '_')
        docstring = test_info.get('docstring', '')
        
        rust_code = f"""#[test]
fn {test_name}() {{
    // {docstring}
    // TODO: Port from NumPy test
    // Original line: {test_info['line']}
    
    // Test implementation here
}}
"""
        return rust_code
    
    def map_numpy_api(self, numpy_call: str) -> Optional[str]:
        """Map NumPy API call to Raptors API"""
        return self.api_mapping.get(numpy_call)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python port_numpy_test.py <numpy_test_file.py>")
        sys.exit(1)
    
    test_file = Path(sys.argv[1])
    if not test_file.exists():
        print(f"Error: File {test_file} not found")
        sys.exit(1)
    
    # Parse NumPy test file
    parser = NumPyTestParser(test_file)
    parser.parse()
    
    # Generate Rust test stubs
    generator = RustTestGenerator()
    
    print(f"Found {len(parser.test_functions)} test functions in {test_file.name}")
    print("\nTest functions:")
    for test_info in parser.test_functions:
        print(f"  - {test_info['name']} (line {test_info['line']})")
        if test_info.get('docstring'):
            print(f"    {test_info['docstring'][:60]}...")
    
    print("\nGenerated Rust test stubs:")
    print("=" * 60)
    for test_info in parser.test_functions:
        print(generator.generate_test_stub(test_info))
        print()


if __name__ == '__main__':
    main()

