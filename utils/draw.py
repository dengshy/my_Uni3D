import plotly.graph_objects as go
import os
import plotly.express as px

def save_point_cloud_html(x, y, z, color, class_name, index, part_data, output_directory="seg_res_html", dataset_name="modelnet"):
    # 确保输入的 x, y, z, color 维度一致
    assert len(x) == len(y) == len(z) == len(color), "Input dimensions must match!"

    # 创建存储目录
    final_directory = os.path.join(output_directory, dataset_name, class_name)
    os.makedirs(final_directory, exist_ok=True)

    # 获取离散颜色序列
    num_classes = len(part_data[class_name])
    discrete_colors = px.colors.qualitative.Set1[:num_classes]

    # 强制设置 colorscale 的范围，即使分类结果只有一类
    if len(set(color)) == 1:  # 如果只有一个分类结果
        single_color = discrete_colors[color[0]]
        colorscale = [[0, single_color], [1, single_color]]  # 强制一个单色的 colorscale
    else:
        colorscale = discrete_colors  # 正常使用多分类的 colorscale

    # 创建图例的 HTML 片段
    legend_html = """
    <div style='position:absolute;bottom:10px;left:10px;padding:10px;background-color:white;border:1px solid black;'>
        <h3>Prompt</h3>
        <ul>
    """
    for idx, label in enumerate(part_data[class_name]):
        legend_html += f"""
        <li>
            <span style='display:inline-block;width:20px;height:20px;background-color:{discrete_colors[idx]};'></span>
            {label}
        </li>
        """
    legend_html += "</ul></div>"

    # 创建 3D 点云图
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,  # 点的坐标
        mode='markers',  # 以点的形式展示
        marker=dict(
            size=3,  # 点的大小
            color=color,  # 映射后的颜色（使用分类索引）
            colorscale=colorscale,  # 设置颜色映射与图例一致
            cmin=0,  # 最小分类索引
            cmax=num_classes - 1,  # 最大分类索引
            opacity=0.8  # 点的透明度
        )
    )])

    # 设置图的布局
    fig.update_layout(
        title=f"3D Point Cloud: {class_name} {index}"
    )

    # 生成 HTML 文件内容
    html_content = fig.to_html(full_html=True)

    # 将图例添加到 HTML 内容中
    html_with_legend = html_content.replace("</body>", f"{legend_html}</body>")

    # 保存到 HTML 文件
    output_path = os.path.join(final_directory, f"{class_name}{index}.html")
    with open(output_path, "w") as f:
        f.write(html_with_legend)

    # 打印保存路径
    print(f"Saved 3D point cloud to: {output_path}")
