import os

# Change this if needed
root_directory = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Yoga Pose Classification"

output_file = "file_tree.txt"

def generate_file_tree(start_path, file):
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = '│   ' * level
        file.write(f"{indent}├── {os.path.basename(root)}\n")

        sub_indent = '│   ' * (level + 1)
        for f in files:
            file.write(f"{sub_indent}├── {f}\n")

with open(output_file, "w", encoding="utf-8") as f:
    generate_file_tree(root_directory, f)

print(f"File tree saved successfully in '{output_file}'")
