import unittest
import bn

class BayesTest(unittest.TestCase):
    """
    Test the bayes network
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_instantiation(self):
        b = bn.make_earthquake_model()
        self.assertTrue(bool(b), "Model failed creation step!")
