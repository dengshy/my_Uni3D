import os
import json

def generate_html_viewer(root_dir, output_html="index.html"):
    """
    遍历目录结构并生成一个 HTML 文件，用于动态查看文件。
    
    :param root_dir: str, 数据集的根目录（包含多个 dataset 文件夹）。
    :param output_html: str, 输出的主 HTML 文件路径。
    """
    # 遍历文件结构
    data_structure = {}
    for dataset in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset)
        if os.path.isdir(dataset_path):
            data_structure[dataset] = {}
            for class_name in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, class_name)
                if os.path.isdir(class_path):
                    data_structure[dataset][class_name] = []
                    for file_name in os.listdir(class_path):
                        if file_name.endswith(".html"):
                            data_structure[dataset][class_name].append(file_name)

    # 将目录结构转换为 JSON 字符串
    json_data = json.dumps(data_structure)

    # 生成 HTML 文件
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HTML File Viewer</title>
    </head>
    <body>
        <h1 style="text-align: center;">HTML File Viewer</h1>
        <div style="text-align: center; margin-bottom: 20px;">
            <label for="dataset" style="font-size: 1.5em;">Dataset:</label>
            <select id="dataset" style="font-size: 1.2em;">
                <option value="">-- Select Dataset --</option>
            </select>
            
            <label for="class" style="font-size: 1.5em; margin-left: 20px;">Class:</label>
            <select id="class" style="font-size: 1.2em;">
                <option value="">-- Select Class --</option>
            </select>
            
            <label for="file" style="font-size: 1.5em; margin-left: 20px;">File:</label>
            <select id="file" style="font-size: 1.2em;">
                <option value="">-- Select File --</option>
            </select>
        </div>

        <!-- iframe 用于显示选定的 HTML 文件 -->
        <iframe id="htmlViewer" style="width: 100%; height: 90vh; border: none;"></iframe>

        <script>
            // 数据结构
            const data = {json_data};

            // 获取筛选框和 iframe 元素
            const datasetSelect = document.getElementById("dataset");
            const classSelect = document.getElementById("class");
            const fileSelect = document.getElementById("file");
            const htmlViewer = document.getElementById("htmlViewer");

            // 初始化 Dataset 下拉框
            Object.keys(data).forEach(dataset => {{
                const option = document.createElement("option");
                option.value = dataset;
                option.textContent = dataset;
                datasetSelect.appendChild(option);
            }});

            // 更新 Class 下拉框
            datasetSelect.addEventListener("change", () => {{
                classSelect.innerHTML = '<option value="">-- Select Class --</option>';
                fileSelect.innerHTML = '<option value="">-- Select File --</option>';
                htmlViewer.src = ""; // 清空 iframe

                const selectedDataset = datasetSelect.value;
                if (selectedDataset && data[selectedDataset]) {{
                    Object.keys(data[selectedDataset]).forEach(className => {{
                        const option = document.createElement("option");
                        option.value = className;
                        option.textContent = className;
                        classSelect.appendChild(option);
                    }});
                }}
            }});

            // 更新 File 下拉框
            classSelect.addEventListener("change", () => {{
                fileSelect.innerHTML = '<option value="">-- Select File --</option>';
                htmlViewer.src = ""; // 清空 iframe

                const selectedDataset = datasetSelect.value;
                const selectedClass = classSelect.value;
                if (selectedClass && data[selectedDataset][selectedClass]) {{
                    data[selectedDataset][selectedClass].forEach(fileName => {{
                        const option = document.createElement("option");
                        option.value = fileName;
                        option.textContent = fileName;
                        fileSelect.appendChild(option);
                    }});
                }}
            }});

            // 更新 iframe 内容
            fileSelect.addEventListener("change", () => {{
                const selectedDataset = datasetSelect.value;
                const selectedClass = classSelect.value;
                const selectedFile = fileSelect.value;

                if (selectedFile) {{
                    const filePath = `{root_dir}/` + selectedDataset + "/" + selectedClass + "/" + selectedFile;
                    htmlViewer.src = filePath;
                }}
            }});
        </script>
    </body>
    </html>
    """

    # 保存 HTML 文件
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"HTML Viewer saved to {output_html}")

if __name__ == "__main__":
    generate_html_viewer("seg_res_html")
