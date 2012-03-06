import aipy_cuda.adder as a
import numpy as n, unittest

class Test_Aipy(unittest.TestCase):
    def test_cuda_add_basics(self):
        x,y = n.arange(10), n.arange(10)
        z = a.cuda_add(x,y)
        self.assertTrue(n.all(z == n.arange(0,20,2)))

if __name__ == '__main__':
    unittest.main()
