import unittest
import argparse
from pathlib import Path
from main import Main

def defaultTestArguments():
    parameters = argparse.Namespace()
    parameters.arffFilesPaths = [
        Path("./datasets/adult.arff"),
        Path("./datasets/vote.arff"),
        Path("./datasets/waveform.arff")
    ]
    parameters.verbose = False
    return parameters

class myTestCase(unittest.TestCase):
    def test_main(self):
        parameters = defaultTestArguments()
        try:
            ret = Main(parameters)()
        except Exception as _:
            assert False

if __name__ == "__main__":
    unittest.main()
