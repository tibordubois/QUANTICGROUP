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

        bn2=gum.fastBN("A->B[9];C->B;D->B")
        qbn2=qBayesNet(bn2)
        width = qbn.getWidth("B")
        self.assertEqual(width,4)
        return

    # Ajoutez d'autres méthodes de test pour tester les autres méthodes de qBayesNet

    
    def test_mapNodeToQBit(self):
        bn=gum.fastBN("A->B<-C->D->E<-F<-A;C->G<-H<-I->J")
        qbn=qBayesNet(bn)
        map=qbn.mapNodeToQBit(qbn.target_nodes)
        for node in bn.nodes()
            self.assertEqual(getWidth(node),len(map[node]))
        return


    #def test_getTotNumQBits     ON N A PAS ENCORE UTILISE CETTE FONCTION ON NE L'UTILISERA SUREMENT PAS

    def test_getBinarizedParameters(self): #Teste aussi getRootNodes et getAllParentStates
        bn=gum.fastBN("A->B<-C->D->E<-F<-A;C->G<-H<-I->J")
        qbn=qBayesNet(bn)
    
        root_nodes = qbn.getRootNodes()
        internal_nodes = qbn.bn.nodes().difference(root_nodes)
        
        for n_id in internal_nodes.intersection(qbn.target_nodes):

            parent_id_set = qbn.bn.parents(n_id).intersection(qbn.target_nodes)
            parent_qbit_list = list(np.ravel([qbn.n_qb_map[p_id] 
                                              for p_id in parent_id_set])) 
            

            for params_dict in qbn.getAllParentSates(n_id): 

                width_dict = {p_id: qbn.getWidth(p_id) for p_id in parent_id_set}
                bin_params = qbn.getBinarizedParameters(width_dict, params_dict)
                self.assertEqual(qbn.getWidth(n_id),len(bin_params[n_id])
    return

    
if __name__ == '__main__':
    unittest.main()