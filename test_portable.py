"""Fast inference checks; expensive score validation is run separately."""
import unittest
import numpy as np
import torch
from hurdle_gamma_copula import PretrainedGenerator

class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.m=PretrainedGenerator()

    def test_decoder_marginals_and_repeatability(self):
        z=self.m.frozen_latents('V0')[:4]
        a=self.m.decode(z);b=self.m.decode(z)
        np.testing.assert_array_equal(a.toarray(),b.toarray())
        self.assertEqual(a.shape,(4,32285))
        self.assertTrue(np.isfinite(a.data).all() and (a.data>=0).all())
        np.testing.assert_allclose(np.expm1(a.toarray()).sum(1),10000,rtol=1e-5)

    def test_gene_order_and_latent_validation(self):
        with self.assertRaises(ValueError):self.m.encode(np.zeros((1,32285)),genes=self.m.genes[::-1])
        with self.assertRaises(ValueError):self.m.decode(np.zeros((2,16)))
        with self.assertRaises(ValueError):self.m.decode(np.full((2,32),np.nan))

    def test_encoder_and_copula_normalization(self):
        x=self.m.decode(self.m.frozen_latents()[:3])
        mu,lv=self.m.encode(x,genes=self.m.genes)
        self.assertEqual(mu.shape,(3,32));self.assertTrue(np.isfinite(lv).all())
        np.testing.assert_allclose((self.m.b**2).sum(1)+self.m.noise**2,1,atol=2e-7)

if __name__=='__main__':unittest.main()
