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
    return parameters

class myTestCase(unittest.TestCase):
    def test_main(self):
        parameters = defaultTestArguments()
        ret = Main(parameters)()
        self.assertEqual(ret, None)

if __name__ == "__main__":
    unittest.main()
