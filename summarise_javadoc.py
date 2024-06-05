import os
import json
import requests
import sys
def process_html_files(folder_path):
    print("Passed folder:", folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as html_file:
                    html_content = html_file.read()

                # Prepare the data payload
                data_payload = {
                    "model": "llama3",
                    "prompt": "Provide human readable summary of this javadoc html file. Information must include information about package it belongs to, or if it is a class or interface it needs to include fully qualified name of it, all information about methods, inhertiance, their descriptions, and any other insights from the file. Summary needs to be suitable for Retrieval Augmented Generation. It should not include any html keywords, use human readable formet. Here is the html file: " + html_content,
                    "stream": False
                }

                # Make the API request
                print("Processing file:",file_path)
                response = requests.post("http://localhost:11434/api/generate", json=data_payload)

                if response.status_code == 200:
                    response_json = response.json()
                    response_text = response_json.get("response", "")

                    # Save the response to a .txt file
                    txt_file_path = os.path.splitext(file_path)[0] + ".txt"
                    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(response_text)
                else:
                    print(f"Failed to process {file_path}: {response.status_code} - {response.text}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <source_folder>")
        sys.exit(1)
    source_folder = sys.argv[1]
    print("Calling...")
    process_html_files(source_folder)

