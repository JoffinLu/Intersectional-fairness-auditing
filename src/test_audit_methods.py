"""Deterministic and exhaustive checks of the inferential claims."""
import itertools
import unittest

import numpy as np
from scipy.stats import binom

from audit_methods import (
    discovery_weights, fixed_prior_mse, fixed_prior_screen_power, four_cell_dirichlet_mean,
    hoeffding_difference_pvalues, posterior_mean,
    reference_upper_binomial_pvalues, true_disparities, weighted_holm,
)


class AuditTests(unittest.TestCase):
    def test_zero_counts_are_inconclusive(self):
        for func in (hoeffding_difference_pvalues, reference_upper_binomial_pvalues):
            out = func([0, 1], [0, 5])
            self.assertEqual(out[0], 1.)
        self.assertFalse(weighted_holm(np.ones(8)).any())

    def test_dirichlet_conditioning_uses_two_cell_prior(self):
        np.testing.assert_allclose(four_cell_dirichlet_mean([0,2],[10,10]),[1/12,3/12])

    def test_uniform_weights_match_standard_holm(self):
        p = np.array([.001, .018, .019, .9])
        np.testing.assert_array_equal(weighted_holm(p), [True, False, False, False])
        np.testing.assert_array_equal(weighted_holm(p, np.ones(4)), weighted_holm(p))

    def test_exact_mse_matches_enumeration(self):
        p, n, mu, kappa = .35, 20, .12, 100
        k = np.arange(n+1)
        probs = binom.pmf(k, n, p)
        naive, shrink = fixed_prior_mse(p, n, mu, kappa)
        self.assertAlmostEqual(np.sum(probs*(k/n-p)**2), naive, places=12)
        self.assertAlmostEqual(np.sum(probs*(posterior_mean(k, np.full_like(k, n), mu, kappa)-p)**2), shrink, places=12)

    def test_exact_screen_boundary(self):
        n, p, mu, kappa, t = 20, .35, .12, 100, .2
        k = np.arange(n+1)
        expected = np.sum(binom.pmf(k,n,p)*(posterior_mean(k,np.full_like(k,n),mu,kappa)>t))
        self.assertAlmostEqual(float(fixed_prior_screen_power(p,n,mu,kappa,t)), expected, places=12)
        self.assertEqual(float(fixed_prior_screen_power(.5, 10, .01, 1000, .2)), 0.)
        self.assertAlmostEqual(float(fixed_prior_screen_power(.4,1,.1,1,.55-1e-13)),.4)

    def test_weights_use_no_confirmation_or_truth(self):
        for mode in ("uniform", "size", "raw", "eb", "conflict"):
            w = discovery_weights([1, 3, 7], [10, 10, 10], mode=mode)
            self.assertAlmostEqual(w.sum(), 1.)
            self.assertTrue(np.all(w >= .1/3-1e-12))

    def test_complement_truth_is_not_generating_increment(self):
        delta = true_disparities([.4,.1,.1], [20,40,40])
        self.assertAlmostEqual(delta[0], .3)
        self.assertAlmostEqual(delta[1], -.1)

    def test_exhaustive_null_superuniformity(self):
        # Small finite support, including unequal complement component rates.
        # Exhaustive probability summation; no Monte Carlo tolerance is used.
        ns = np.array([8, 5, 7])
        states = np.array(list(itertools.product(*(range(v+1) for v in ns))))
        for ps in ([.15,.1,.2], [.25,.2,.2], [.05,.01,.1]):
            ps = np.asarray(ps)
            masses = np.prod(binom.pmf(states, ns, ps), axis=1)
            truth = true_disparities(ps, ns)
            for func in (hoeffding_difference_pvalues, reference_upper_binomial_pvalues):
                values = func(states, ns, margin=.05)
                for a in (.01,.05,.1,.2,.5):
                    for g in np.flatnonzero(truth <= .05+1e-12):
                        self.assertLessEqual(float(masses[values[:,g] <= a].sum()), a+1e-10)
                    rejected = weighted_holm(values, alpha=a)
                    bad = np.any(rejected[:, truth <= .05+1e-12], axis=1)
                    self.assertLessEqual(float(masses[bad].sum()), a+1e-10)
                    nonuniform = weighted_holm(values, weights=[.6,.3,.1], alpha=a)
                    bad_weighted = np.any(nonuniform[:, truth <= .05+1e-12], axis=1)
                    self.assertLessEqual(float(masses[bad_weighted].sum()), a+1e-10)


if __name__ == "__main__":
    unittest.main()
