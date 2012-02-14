import aipy_cuda._aipy as a
import numpy as n, unittest

class Test_Aipy(unittest.TestCase):
    def test_cuda_add_basics(self):
        a,b = n.arange(10), n.arange(10)
        c = a.cuda_add(a,b)
        self.assertTrue(n.all(c == n.arange(0,20,2)))

if __name__ == '__main__':
    unittest.main()
