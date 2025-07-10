import os
import ast

def extract_imports(path):
    imports = set()
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    try:
                        node = ast.parse(f.read())
                        for n in ast.walk(node):
                            if isinstance(n, ast.Import):
                                for alias in n.names:
                                    imports.add(alias.name.split('.')[0])
                            elif isinstance(n, ast.ImportFrom):
                                if n.module:
                                    imports.add(n.module.split('.')[0])
                    except:
                        pass
    return sorted(list(imports))

import pkg_resources
wanted = extract_imports('.')
print(wanted)
for pkg in pkg_resources.working_set:
    if pkg.key in wanted:
        print(f"{pkg.project_name}=={pkg.version}")

