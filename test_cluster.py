import sklearn.cluster
import sklearn.datasets
import numpy as np
import pandas as pd
import time

import cudf
import cuml


def benchmark_algorithm(
    dataset_sizes,
    cluster_function,
    function_args,
    function_kwds,
    dataset_dimension=10,
    dataset_n_clusters=10,
    max_time=45,
    sample_size=2,
):
    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result = np.nan * np.ones((len(dataset_sizes), sample_size))
    for index, size in enumerate(dataset_sizes):
        for s in range(sample_size):
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            data, labels = sklearn.datasets.make_blobs(
                n_samples=size, n_features=dataset_dimension, centers=dataset_n_clusters
            )

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time

            # If we are taking more than max_time then abort -- we don't
            # want to spend excessive time on slow algorithms
            if time_taken > max_time:
                result[index, s] = time_taken
                return pd.DataFrame(
                    np.vstack([dataset_sizes.repeat(sample_size), result.flatten()]).T,
                    columns=["x", "y"],
                )
            else:
                result[index, s] = time_taken

    # Return the result as a dataframe for easier handling with seaborn afterwards
    return pd.DataFrame(
        np.vstack([dataset_sizes.repeat(sample_size), result.flatten()]).T,
        columns=["x", "y"],
    )


def benchmark(dataset_size, dataset_dimension, n_clusters):
    huge_dataset_sizes = np.arange(1, 41) * dataset_size
    k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
    kmeans_float = cuml.KMeans(n_clusters=n_clusters)

    huge_cuml_data = benchmark_algorithm(
        huge_dataset_sizes,
        kmeans_float.fit,
        (),
        {},
        max_time=240,
        sample_size=2,
        dataset_dimension=dataset_dimension,
    )
    huge_k_means_data = benchmark_algorithm(
        huge_dataset_sizes,
        k_means.fit,
        (),
        {},
        max_time=240,
        sample_size=2,
        dataset_dimension=dataset_dimension,
    )
    results = pd.merge(
        huge_cuml_data,
        huge_k_means_data,
        on="x",
        how="left",
        suffixes=("_cuml", "_sklearn"),
    )
    results = results.groupby("x").agg(np.mean)
    return results


if __name__ == "__main__":
    dataset_size = 5000
    dataset_dimension = 100
    n_clusters = 10
    bench_results = benchmark(dataset_size, dataset_dimension, n_clusters)
    print(bench_results)
    bench_results.to_csv("bench_cluster.csv")
