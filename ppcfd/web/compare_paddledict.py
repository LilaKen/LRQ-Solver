import paddle
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np

def read_paddledict(file_path):
    """
    读取一个 .paddledict 文件，并返回其中的内容。

    Args:
        file_path (str): .paddledict 文件的路径。

    Returns:
        dict: 文件中的内容。
    """
    try:
        data = paddle.load(file_path)
        if not isinstance(data, dict):
            raise ValueError(f"Loaded data from {file_path} is not a dictionary.")
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # 定义 .paddledict 文件的路径
    file1_path = './EP35_MCE_0a_T06_0219@6000.case.paddledict'
    file2_path = './output/2025-04-08-12-39-06/EP35_MCE_0a_T06_0219@6000_data.paddledict'

    # 读取文件
    data1 = read_paddledict(file1_path)
    data2 = read_paddledict(file2_path)

    query_points = data1["centroids"]
    reference_points = data2["centroids"]

    tree = cKDTree(query_points)
    _, indices = tree.query(reference_points, k=1)

    print("The maximum L2 error of centroid: ")
    print(paddle.norm(query_points[indices] - reference_points, axis=-1).max().numpy())

    area_1 = data1["areas"][indices]
    normal_1 = data1["normal"][indices]

    area_2 = data2["areas"]
    normal_2 = data2["normal"]

    print("The maximum L2 error of area: ")
    print(paddle.norm(area_1 - area_2, axis=-1).max().numpy())
    print((paddle.norm(area_1 - area_2, axis=-1) / paddle.norm(area_1, axis=-1)).max().numpy()*100)
    print("The maximum L2 error of normal: ")
    print(paddle.norm(normal_1 - normal_2, axis=-1).max().numpy())
    print((paddle.norm(normal_1 - normal_2, axis=-1) / paddle.norm(normal_1, axis=-1)).max().numpy()*100)

    normal_error = paddle.sum(normal_1 * normal_2, axis=-1).numpy().flatten()
    print((np.abs(normal_error) > 0.9999).sum() / normal_error.shape[0])
    plt.figure(dpi=300)
    plt.hist(normal_error, bins=100)
    plt.title("The cosine similarity of surface normal vertor")
    plt.savefig("./normal_error.png")

if __name__ == "__main__":
    main()