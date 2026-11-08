"""Meaningful checks of interval mathematics and small-support coverage."""
import itertools
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta, binom
from interval_diagnostics import wald_intervals, saft_style_intervals, cp_simultaneous_intervals
from audit_methods import true_disparities


class IntervalChecks(unittest.TestCase):
    def test_wald_matches_multinomial_delta_variance(self):
        n = np.array([20, 80])
        k = np.array([7, 9])
        joint = np.array([.2*.35, .8*.1125, .2, .8])
        # Covariance of (E*I_g, E*I_c, I_g, I_c) from four atomic outcomes.
        atoms = np.array([[0,0,1,0],[1,0,1,0],[0,0,0,1],[0,1,0,1]],float)
        mass = np.array([.2*.65,.2*.35,.8*.8875,.8*.1125])
        covariance = (atoms*mass[:,None]).T@atoms - np.outer(joint,joint)
        gradient = np.array([1/.2,-1/.8,-joint[0]/.2**2,joint[1]/.8**2])
        se = np.sqrt(gradient@covariance@gradient/n.sum())
        lo,hi = wald_intervals(k,n)
        from scipy.stats import norm
        self.assertAlmostEqual((hi[0]-lo[0])/(2*norm.ppf(.975)),se,places=13)

    def test_posterior_quantiles_against_independent_quadrature(self):
        k,n = np.array([4,19]),np.array([20,100])
        lo,hi=saft_style_intervals(k,n,125,draws=100000)
        def cdf(z):
            return quad(lambda r: beta.cdf(np.clip(z+r,0,1),5,17)*beta.pdf(r,20,82),0,1,epsabs=1e-10)[0]
        self.assertLess(abs(cdf(lo[0])-.025),.003)
        self.assertLess(abs(cdf(hi[0])-.975),.003)

    def test_exact_rectangle_coverage_on_enumerated_small_families(self):
        n=np.array([3,4])
        counts=np.array(list(itertools.product(range(4),range(5))))
        lo,hi=cp_simultaneous_intervals(counts,n,alpha=.2)
        for p in itertools.product([0,.01,.2,.5,.8,.99,1],repeat=2):
            truth=true_disparities(np.array(p),n)
            mass=binom.pmf(counts[:,0],n[0],p[0])*binom.pmf(counts[:,1],n[1],p[1])
            covered=((lo<=truth)&(truth<=hi)).all(axis=1)
            self.assertGreaterEqual(float(mass[covered].sum()),.8-1e-12)

    def test_insufficient_support_stays_inconclusive(self):
        for function in (wald_intervals,cp_simultaneous_intervals):
            lo,hi=function(np.array([0,4]),np.array([0,10]))
            self.assertTrue(np.all(np.isneginf(lo)))
            self.assertTrue(np.all(np.isposinf(hi)))

    def test_switch_checks_four_outcome_counts(self):
        n=np.array([200,1000])
        k=np.array([80,100])
        expected=wald_intervals(k,n)
        actual=saft_style_intervals(k,n,82)
        np.testing.assert_array_equal(actual[0],expected[0])
        k=np.array([10,100])
        wald=wald_intervals(k,n)
        bayes=saft_style_intervals(k,n,82)
        self.assertGreater(np.max(np.abs(bayes[0]-wald[0])),1e-4)


if __name__=='__main__':
    unittest.main()
