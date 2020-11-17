import argparse
from pathlib import Path


def parseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser("")
    return parser.parse_args()


class Main:
    def __init__(self, args):
        pass

    def __call__(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    args = parseArgumentsFromCommandLine()
    Main(args)()
