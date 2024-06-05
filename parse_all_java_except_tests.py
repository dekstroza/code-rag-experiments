import os
import javalang
import json
import re
import argparse

def extract_javadoc_tags(javadoc):
    tags = {'author': '', 'since': '', 'see': []}
    if javadoc:
        author_match = re.search(r'@author\s+([^\n]+)', javadoc)
        since_match = re.search(r'@since\s+([^\n]+)', javadoc)
        see_matches = re.findall(r'@see\s+([^\n]+)', javadoc)

        if author_match:
            tags['author'] = author_match.group(1).strip()
        if since_match:
            tags['since'] = since_match.group(1).strip()
        if see_matches:
            tags['see'] = [match.strip() for match in see_matches]

    return tags

def parse_java_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()

    tree = javalang.parse.parse(java_code)
    class_info = {}

    for path, node in tree:
        if isinstance(node, javalang.tree.PackageDeclaration):
            class_info['package'] = node.name

        if isinstance(node, javalang.tree.ClassDeclaration):
            class_info['name'] = node.name
            javadoc_tags = extract_javadoc_tags(node.documentation)
            class_info['author'] = javadoc_tags.get('author', '')
            class_info['since'] = javadoc_tags.get('since', '')
            class_info['see'] = javadoc_tags.get('see', [])
            class_info['description'] = node.documentation if node.documentation else ""
            class_info['methods'] = []

            for method in node.methods:
                method_info = {
                    'name': method.name,
                    'parameters': [
                        {'name': param.name, 'type': param.type.name}
                        for param in method.parameters
                    ],
                    'returnType': method.return_type.name if method.return_type else 'void',
                    'description': method.documentation if method.documentation else ""
                }
                class_info['methods'].append(method_info)
    return class_info

def generate_json_description(file_path):
    class_info = parse_java_file(file_path)
    return json.dumps(class_info, indent=2)

def process_java_file(java_file_path):
    if("src/main/java" in java_file_path):
        json_description = generate_json_description(java_file_path)
        json_file_path = os.path.splitext(java_file_path)[0] + '.json'
        print("Processed:", java_file_path)
        with open(json_file_path, 'w') as json_file:
            json_file.write(json_description)

def main():
    parser = argparse.ArgumentParser(description="Generate JSON descriptions of Java classes in a directory.")
    parser.add_argument('file_path', help="Path to the Java file or directory containing Java files.")

    args = parser.parse_args()

    if os.path.isdir(args.file_path):
        for root, _, files in os.walk(args.file_path):
            for file_name in files:
                if file_name.endswith('.java'):
                    java_file_path = os.path.join(root, file_name)
                    process_java_file(java_file_path)
    elif os.path.isfile(args.file_path) and args.file_path.endswith('.java'):
        process_java_file(args.file_path)
    else:
        print("Invalid input. Please provide a valid Java file or directory.")

if __name__ == "__main__":
    main()

