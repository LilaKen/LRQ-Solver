import os
import threading

from paddle.utils.cpp_extension import load

import ppcfd.paddle_custom_operator.return_types as return_types

dir_path = os.path.dirname(os.path.abspath(__file__)) + "/"
open3d_path = [
    dir_path + "../../Open3D/cpp/",
    dir_path + "../../Open3D/",
    dir_path + "../../eigen/",
]


class CustomOps:
    # Singleton pattern
    # Thread lock
    _instance_lock = threading.Lock()
    # [CustomOps.instance] is before [CustomOps.__init__]
    @classmethod
    def instance(cls, *args, **kwargs):
        # Serial execute, because [paddle.utils.cpp_extension.load()] is not thread safe
        with CustomOps._instance_lock:
            if not hasattr(CustomOps, "_instance"):
                CustomOps._instance = CustomOps(*args, **kwargs)
            return CustomOps._instance

    def __init__(self):
        # compiling custom operator
        # Op 1 : fixed_radius_search
        sources = [
            "FixedRadiusSearchOps.cpp",
            "FixedRadiusSearchOpKernel.cu",
            "FixedRadiusSearchOpKernel.cpp",
        ]
        sources = [dir_path + f for f in sources]
        self.ops = {}
        self.ops["fixed_radius_search"] = load(
            name="open_3d_fixed_radius_search",
            sources=sources,
            extra_ldflags=["-ltbb"],
            extra_cxx_cflags=["-DBUILD_CUDA_MODULE"],
            extra_include_paths=open3d_path,
        )

        # Op 2: build_spatial_hash_table
        sources = [
            "BuildSpatialHashTableOps.cpp",
            "BuildSpatialHashTableOpKernel.cu",
            "BuildSpatialHashTableOpKernel.cpp",
        ]
        sources = [dir_path + f for f in sources]
        self.ops["build_spatial_hash_table"] = load(
            name="open_3d_build_spatial_hash_table",
            sources=sources,
            extra_ldflags=["-ltbb"],
            extra_cxx_cflags=["-DBUILD_CUDA_MODULE"],
            extra_include_paths=open3d_path,
        )


def fixed_radius_search(
    points,
    queries,
    radius,
    points_row_splits,
    queries_row_splits,
    hash_table_splits,
    hash_table_index,
    hash_table_cell_splits,
    index_dtype=3,
    metric="L2",
    ignore_query_point=False,
    return_distances=False,
):
    current_ops = CustomOps.instance().ops["fixed_radius_search"]
    tmp = current_ops.open_3d_fixed_radius_search(
        points=points,
        queries=queries,
        points_row_splits=points_row_splits,
        queries_row_splits_gpu=queries_row_splits,
        hash_table_splits_gpu=hash_table_splits,
        hash_table_index=hash_table_index,
        hash_table_cell_splits=hash_table_cell_splits,
        index_dtype=int(index_dtype),
        radius=radius,
        metric_str=metric,
        ignore_query_point=ignore_query_point,
        return_distances=return_distances,
    )
    return return_types.fixed_radius_search(*tmp)


def build_spatial_hash_table(
    points,
    radius,
    points_row_splits,
    hash_table_size_factor,
    max_hash_table_size=33554432,
):
    """Creates a spatial hash table meant as input for fixed_radius_search


    The following example shows how **build_spatial_hash_table** and
    **fixed_radius_search** are used together::

    import open3d.ml.tf as ml3d

    points = [
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]]

    queries = [
        [1.0,1.0,1.0],
        [0.5,2.0,2.0],
        [0.5,2.1,2.1],
    ]

    radius = 1.0

    # build the spatial hash table for fixex_radius_search
    table = ml3d.ops.build_spatial_hash_table(points,
                                                radius,
                                                points_row_splits=paddle.LongTensor([0,5]),
                                                hash_table_size_factor=1/32)

    # now run the fixed radius search
    ml3d.ops.fixed_radius_search(points,
                                queries,
                                radius,
                                points_row_splits=paddle.LongTensor([0,5]),
                                queries_row_splits=paddle.LongTensor([0,3]),
                                **table._asdict())
    # returns neighbors_index      = [1, 4, 4]
    #         neighbors_row_splits = [0, 1, 2, 3]
    #         neighbors_distance   = []

    # or with paddle
    import paddle
    import open3d.ml.paddle as ml3d

    points = paddle.Tensor([
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]])

    queries = paddle.Tensor([
        [1.0,1.0,1.0],
        [0.5,2.0,2.0],
        [0.5,2.1,2.1],
    ])

    radius = 1.0

    # build the spatial hash table for fixex_radius_search
    table = ml3d.ops.build_spatial_hash_table(points,
                                                radius,
                                                points_row_splits=paddle.LongTensor([0,5]),
                                                hash_table_size_factor=1/32)

    # now run the fixed radius search
    ml3d.ops.fixed_radius_search(points,
                                queries,
                                radius,
                                points_row_splits=paddle.LongTensor([0,5]),
                                queries_row_splits=paddle.LongTensor([0,3]),
                                **table._asdict())
    # returns neighbors_index      = [1, 4, 4]
    #         neighbors_row_splits = [0, 1, 2, 3]
    #         neighbors_distance   = []



    max_hash_table_size: The maximum hash table size.

    points: The 3D positions of the input points.

    radius: A scalar which defines the spatial cell size of the hash table.

    points_row_splits: 1D vector with the row splits information if points is
    batched. This vector is [0, num_points] if there is only 1 batch item.

    hash_table_size_factor:
    The size of the hash table as a factor of the number of input points.

    hash_table_index: Stores the values of the hash table, which are the indices of
    the points. The start and end of each cell is defined by
    **hash_table_cell_splits**.

    hash_table_cell_splits: Defines the start and end of each hash table cell within
    a hash table.

    hash_table_splits: Defines the start and end of each hash table in the
    hash_table_cell_splits array. If the batch size is 1 then there is only one
    hash table and this vector is [0, number of cells].
    """
    current_ops = CustomOps.instance().ops["build_spatial_hash_table"]
    tmp = current_ops.open_3d_build_spatial_hash_table(
        points=points,
        points_row_splits=points_row_splits,
        radius=radius,
        hash_table_size_factor=hash_table_size_factor,
        max_hash_table_size=max_hash_table_size,
    )

    return return_types.build_spatial_hash_table(*tmp)


def radius_search(
    points,
    queries,
    radii,
    points_row_splits,
    queries_row_splits,
    index_dtype=3,
    metric="L2",
    ignore_query_point=False,
    return_distances=False,
    normalize_distances=False,
):
    """Computes the indices and distances of all neighbours within a radius.

    This op computes the neighborhood for each query point and returns the indices
    of the neighbors and optionally also the distances. Each query point has an
    individual search radius. Points and queries can be batched with each batch
    item having an individual number of points and queries. The following example
    shows a simple search with just a single batch item::

      import open3d.ml.tf as ml3d

      points = [
          [0.1,0.1,0.1],
          [0.5,0.5,0.5],
          [1.7,1.7,1.7],
          [1.8,1.8,1.8],
          [0.3,2.4,1.4]]

      queries = [
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.2],
      ]

      radii = [1.0,1.0,1.0]

      ml3d.ops.radius_search(points, queries, radii,
                             points_row_splits=[0,5],
                             queries_row_splits=[0,3])
      # returns neighbors_index      = [1, 4, 4]
      #         neighbors_row_splits = [0, 1, 2, 3]
      #         neighbors_distance   = []


      # or with paddle
      import paddle
      import open3d.ml.paddle as ml3d

      points = paddle.Tensor([
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]])

      queries = paddle.Tensor([
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.1],
      ])

      radii = paddle.Tensor([1.0,1.0,1.0])

      ml3d.ops.radius_search(points, queries, radii,
                             points_row_splits=paddle.LongTensor([0,5]),
                             queries_row_splits=paddle.LongTensor([0,3]))
      # returns neighbors_index      = [1, 4, 4]
      #         neighbors_row_splits = [0, 1, 2, 3]
      #         neighbors_distance   = []


    metric: Either L1 or L2. Default is L2

    ignore_query_point: If true the points that coincide with the center of the
      search window will be ignored. This excludes the query point if **queries** and
      **points** are the same point cloud.

    return_distances: If True the distances for each neighbor will be returned in
      the output tensor **neighbors_distance**.  If False a zero length Tensor will
      be returned for **neighbors_distances**.

    normalize_distances: If True the returned distances will be normalized with the
      radii.

    points: The 3D positions of the input points.

    queries: The 3D positions of the query points.

    radii: A vector with the individual radii for each query point.

    points_row_splits: 1D vector with the row splits information if points is
      batched. This vector is [0, num_points] if there is only 1 batch item.

    queries_row_splits: 1D vector with the row splits information if queries is
      batched. This vector is [0, num_queries] if there is only 1 batch item.

    neighbors_index: The compact list of indices of the neighbors. The
      corresponding query point can be inferred from the
      **neighbor_count_row_splits** vector.

    neighbors_row_splits: The exclusive prefix sum of the neighbor count for the
      query points including the total neighbor count as the last element. The
      size of this array is the number of queries + 1.

    neighbors_distance: Stores the distance to each neighbor if **return_distances**
      is True. The distances are squared only if metric is L2.
      This is a zero length Tensor if **return_distances** is False.
    """
    sources = ["RadiusSearchOpKernel.cpp", "RadiusSearchOps.cpp"]
    sources = [dir_path + f for f in sources]
    ops_radius_search = load(
        name="open_3d_fixed_radius_search",
        sources=sources,
        extra_include_paths=open3d_path,
    )

    return return_types.radius_search(
        *ops_radius_search(
            points=points,
            queries=queries,
            radii=radii,
            points_row_splits=points_row_splits,
            queries_row_splits=queries_row_splits,
            index_dtype=index_dtype,
            metric=metric,
            ignore_query_point=ignore_query_point,
            return_distances=return_distances,
            normalize_distances=normalize_distances,
        )
    )


def knn_search(
    points,
    queries,
    k,
    points_row_splits,
    queries_row_splits,
    index_dtype=3,
    metric="L2",
    ignore_query_point=False,
    return_distances=False,
):
    """Computes the indices of k nearest neighbors.

    This op computes the neighborhood for each query point and returns the indices
    of the neighbors. The output format is compatible with the radius_search and
    fixed_radius_search ops and supports returning less than k neighbors if there
    are less than k points or ignore_query_point is enabled and the **queries** and
    **points** arrays are the same point cloud. The following example shows the usual
    case where the outputs can be reshaped to a [num_queries, k] tensor::

      import tensorflow as tf
      import open3d.ml.tf as ml3d

      points = [
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]]

      queries = [
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.2],
      ]

      ans = ml3d.ops.knn_search(points, queries, k=2,
                          points_row_splits=[0,5],
                          queries_row_splits=[0,3],
                          return_distances=True)
      # returns ans.neighbors_index      = [1, 2, 4, 2, 4, 2]
      #         ans.neighbors_row_splits = [0, 2, 4, 6]
      #         ans.neighbors_distance   = [0.75 , 1.47, 0.56, 1.62, 0.77, 1.85]
      # Since there are more than k points and we do not ignore any points we can
      # reshape the output to [num_queries, k] with
      neighbors_index = tf.reshape(ans.neighbors_index, [3,2])
      neighbors_distance = tf.reshape(ans.neighbors_distance, [3,2])


      # or with paddle
      import paddle
      import open3d.ml.paddle as ml3d

      points = paddle.Tensor([
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]])

      queries = paddle.Tensor([
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.2],
      ])

      radii = paddle.Tensor([1.0,1.0,1.0])

      ans = ml3d.ops.knn_search(points, queries, k=2,
                                points_row_splits=paddle.LongTensor([0,5]),
                                queries_row_splits=paddle.LongTensor([0,3]),
                                return_distances=True)
      # returns ans.neighbors_index      = [1, 2, 4, 2, 4, 2]
      #         ans.neighbors_row_splits = [0, 2, 4, 6]
      #         ans.neighbors_distance   = [0.75 , 1.47, 0.56, 1.62, 0.77, 1.85]
      # Since there are more than k points and we do not ignore any points we can
      # reshape the output to [num_queries, k] with
      neighbors_index = ans.neighbors_index.reshape(3,2)
      neighbors_distance = ans.neighbors_distance.reshape(3,2)

    metric: Either L1 or L2. Default is L2

    ignore_query_point: If true the points that coincide with the center of the
      search window will be ignored. This excludes the query point if **queries** and
     **points** are the same point cloud.

    return_distances: If True the distances for each neighbor will be returned in
      the output tensor **neighbors_distances**. If False a zero length Tensor will
      be returned for **neighbors_distances**.

    points: The 3D positions of the input points.

    queries: The 3D positions of the query points.

    k: The number of nearest neighbors to search.

    points_row_splits: 1D vector with the row splits information if points is
      batched. This vector is [0, num_points] if there is only 1 batch item.

    queries_row_splits: 1D vector with the row splits information if queries is
      batched. This vector is [0, num_queries] if there is only 1 batch item.

    neighbors_index: The compact list of indices of the neighbors. The
      corresponding query point can be inferred from the
      **neighbor_count_prefix_sum** vector. Neighbors for the same point are sorted
      with respect to the distance.

      Note that there is no guarantee that there will be exactly k neighbors in some cases.
      These cases are:
        * There are less than k points.
        * **ignore_query_point** is True and there are multiple points with the same position.

    neighbors_row_splits: The exclusive prefix sum of the neighbor count for the
      query points including the total neighbor count as the last element. The
      size of this array is the number of queries + 1.

    neighbors_distance: Stores the distance to each neighbor if **return_distances**
      is True. The distances are squared only if metric is L2. This is a zero length
      Tensor if **return_distances** is False.
    """
    sources = ["KnnSearchOpKernel.cpp", "KnnSearchOps.cpp"]
    sources = [dir_path + f for f in sources]
    ops_knn_search = load(
        name="open3d_knn_search", sources=sources, extra_include_paths=open3d_path
    )
    return return_types.knn_search(
        *ops_knn_search(
            points=points,
            queries=queries,
            k=k,
            points_row_splits=points_row_splits,
            queries_row_splits=queries_row_splits,
            index_dtype=index_dtype,
            metric=metric,
            ignore_query_point=ignore_query_point,
            return_distances=return_distances,
        )
    )
