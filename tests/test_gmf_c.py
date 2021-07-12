import unittest
import numpy as n
import time
from hardtarget.analysis.gmf.gmf_c import gmf

class Test_Gmf_C(unittest.TestCase):

    def test_gmf_c(self):
        #This is what we expect as an output from this test
        expected = {'ri': 500, 
                    'gmf_vec': 1e04, 
                    'v_vec': 0.0, 
                    'a_vec': 0.0}

        z_tx=n.zeros(10000,dtype=n.complex64)
        z_rx=n.zeros(12000,dtype=n.complex64)

        for i in range(10):
            z_tx[(i*1000):(i*1000+20)]=1.0
            z_rx[(i*1000+500):(i*1000+(500+20))]=0.5 # simulated "echo"
        
        dec=10
        acc_phasors=n.zeros([20,1000],dtype=n.complex64)
        acc_phasors[0,:]=1.0
        rgs=n.zeros(1000,dtype=n.float32)#arange(700,dtype=n.int64)

        for ri in range(len(rgs)):
            rgs[ri]=ri

        n_r=len(rgs)
        gmf_vec=n.zeros(n_r,dtype=n.float32);
        gmf_dc_vec=n.zeros(n_r,dtype=n.float32);
        v_vec=n.zeros(n_r,dtype=n.float32);
        a_vec=n.zeros(n_r,dtype=n.float32);

        for i in range(20):
            gmf(z_tx,z_rx,acc_phasors,rgs,dec,gmf_vec,gmf_dc_vec,v_vec,a_vec)

        ri=n.argmax(gmf_vec)

        #Check that the output is as we expect
        self.assertEqual(expected['ri'], ri)
        self.assertEqual(expected['gmf_vec'], gmf_vec[ri])
        self.assertEqual(expected['v_vec'], v_vec[ri])
        self.assertEqual(expected['a_vec'], a_vec[ri])

if __name__=='__main__':
    unittest.main()