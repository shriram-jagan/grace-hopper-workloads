import argparse


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nelements",
        nargs="+",
        type=float,
        default=[5],
        dest="nelements",
        help="Number of elements in millions",
    )

    # p=0 -> fully on the cpu, p=1 -> fully on the gpu
    parser.add_argument(
        "-p",
        "--partitions",
        nargs="+",
        type=float,
        default=[0.5],
        action="store",
        dest="partitions",
        help="A list that represents the fraction of workload that "
        "runs on the GPU. p=0 -> fully on the cpu, p=1 -> fully "
        " on the gpu. Results gathered with single-digit accuracy for "
        " partitions, so please use, say, 0.1 and 0.2 instead of "
        " 0.15 and 0.16.",
    )

    return parser


def parse_args():

    parser = get_parser()
    args, _ = parser.parse_known_args()

    return args
