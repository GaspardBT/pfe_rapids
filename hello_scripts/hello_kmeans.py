import argparse
import time

from sklearn.datasets import make_blobs

from cuml import KMeans


def parse_cmd():
    parser = argparse.ArgumentParser(description="Kmeans cuML example.")
    parser.add_argument(
        "--n_samples", default=10000, type=int, help="Number of sample for the test",
    )
    parser.add_argument(
        "--dataset_dimension",
        default=10,
        type=int,
        help="Number of dimension for the test",
    )
    parser.add_argument(
        "--n_clusters", default=10, type=int, help="Number of clusters for the test",
    )

    args = vars(parser.parse_args())
    return args


def main():
    args = parse_cmd()
    start_time = time.time()
    data, labels = make_blobs(
        n_samples=args["n_samples"],
        n_features=args["dataset_dimension"],
        centers=args["n_clusters"],
    )
    time_creation_dataset = start_time - time.time()
    kmeans = KMeans(n_clusters=args["n_clusters"])
    time_init_model = time_creation_dataset - time.time()
    kmeans.fit(data)
    time_compute = time_init_model - time.time()

    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    times = {
        "time_creation_dataset": time_creation_dataset,
        "time_init_model": time_init_model,
        "time_compute": time_compute,
    }
    return labels, cluster_centers, times


if __name__ == "__main__":

    _, _, times = main()
    print(times)
