import sys
import unittest
import pyAgrum as gum

sys.path.append('..')
from qBN.qBNclass import qBayesNet


class TestQBayesNet(unittest.TestCase):
    def test_getWidth(self):
        # Testez la méthode getWidth
        # Créez une instance de qBayesNet
        bn=gum.fastBN("A->B[8];C->B;D->B")
        qbn = qBayesNet(bn)
        width = qbn.getWidth("A") 
        self.assertEqual(width, 1)
        width = qbn.getWidth("B") 
        self.assertEqual(width, 3)

    # Ajoutez d'autres méthodes de test pour tester les autres méthodes de qBayesNet
    
if __name__ == '__main__':
    unittest.main()