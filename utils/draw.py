import os
import plotly.graph_objects as go

def save_point_cloud_html(x, y, z, color, class_name, index,output_directory = "seg_res_html",dataset_name="modelnet"):
    # 确保输入的 x, y, z, color 维度一致
    assert len(x) == len(y) == len(z) == len(color), "Input dimensions must match!"

    final_directory=os.path.join(output_directory,dataset_name,class_name)
    os.makedirs(final_directory, exist_ok=True)

    # 创建 3D 点云图
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,  # 点的坐标
        mode='markers',  # 以点的形式展示
        marker=dict(
            size=2,  # 点的大小
            color=color,  # 点的颜色
            colorscale='Viridis',  # 配色方案
            opacity=0.8  # 点的透明度
        )
    )])

    # 设置图的布局（可选）
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        title=f"3D Point Cloud: {class_name} {index}"
    )

    # 生成文件名
    output_path = os.path.join(final_directory, f"{class_name}{index}.html")

    # 保存为 HTML 文件
    fig.write_html(output_path)

    # 打印保存路径
    print(f"Saved 3D point cloud to: {output_path}")
