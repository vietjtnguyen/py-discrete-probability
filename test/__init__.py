from math import log
import unittest

from discrete_probability import *

class TestBasics(unittest.TestCase):
    def test_helpers(self):
        self.assertIn(weighted_choose([(0.1, 'A'), (0.4, 'B'), (0.5, 'C')]), ['A', 'B', 'C'])
        with self.assertRaises(ValueError):
            for i in range(0, 100000):
                weighted_choose([(0.5, 'A'), (0.4, 'B')])
            self.fail("You're very unlucky, you couldn't get something with a 10% chance after 100,000 trials.")
    def test_binary_variable(self):
        A = Variable('A')
        self.assertEqual(A.values, (True, False))
        self.assertEqual(A.description, 'A')
        self.assertEqual(SingleAssignment(A, True), A << True)
        self.assertEqual(SingleAssignment(A, False), A << False)
        self.assertEqual(A.assignments, (A << True, A << False))
        self.assertTrue(isinstance(A << False, SingleAssignment))
        self.assertTrue(isinstance(A << True, SingleAssignment))
        with self.assertRaises(ValueError):
            A << 2
    def test_multivalued_variable(self):
        B = Variable('B', (0, 1, 'x', 'y'), 'multiple values')
        self.assertEqual(B.values, (0, 1, 'x', 'y'))
        self.assertEqual(B.description, 'multiple values')
        self.assertEqual(SingleAssignment(B, 0), B << 0)
        self.assertEqual(SingleAssignment(B, 1), B << 1)
        self.assertEqual(SingleAssignment(B, 'x'), B << 'x')
        self.assertEqual(SingleAssignment(B, 'y'), B << 'y')
        self.assertEqual(B.assignments, (B << 0, B << 1, B << 'x', B << 'y'))
        self.assertTrue(isinstance(B << 0, SingleAssignment))
        self.assertTrue(isinstance(B << 1, SingleAssignment))
        self.assertTrue(isinstance(B << 'x', SingleAssignment))
        self.assertTrue(isinstance(B << 'y', SingleAssignment))
        with self.assertRaises(ValueError):
            B << 'a'
    def test_assignment(self):
        A, B, C = map(Variable, 'ABC')
        a, a_ = A
        b, b_ = B
        c, c_ = C
        self.assertEqual(len(Assignment.empty), 0)
        self.assertEqual(Assignment(a), Assignment([a]))
        self.assertTrue(Assignment(b_).consistent_with(Assignment([a_, b_, c])))
        self.assertFalse(Assignment(b_).consistent_with(Assignment([a_, b, c])))
        self.assertTrue(Assignment([a, b_]).consistent_with(Assignment([a, b_, c_])))
        self.assertFalse(Assignment([a, b_]).consistent_with(Assignment([a, b, c_])))
        self.assertEqual(Assignment([a_, b, c_]).project([B, C]), Assignment([b, c_]))
        self.assertEqual(Assignment([a, a_, b, c_]).project([A]), Assignment([a, a_]))
        self.assertEqual(Assignment([a_, b, c_]).get_variable(B), b)
        self.assertEqual(Assignment([a, b, c_]).get_variable(A), a)
        self.assertSetEqual(Assignment([a, b_]).get_variables(), frozenset([A, B]))
        self.assertSetEqual(Assignment([a_, b, c_]).get_variables(), frozenset([A, B, C]))
        self.assertSequenceEqual(Assignment([a, b_, c_]).ordered([B, C, A]), [b_, c_, a])
        self.assertSequenceEqual(Assignment([a, b_, c_]).ordered_values([B, C, A]), [False, False, True])
        assignments = Assignment(a).complete([B, C])
        self.assertIn(Assignment([a, b , c ]), assignments)
        self.assertIn(Assignment([a, b , c_]), assignments)
        self.assertIn(Assignment([a, b_, c ]), assignments)
        self.assertIn(Assignment([a, b_, c_]), assignments)
        self.assertNotIn(Assignment([a_, b , c ]), assignments)
        self.assertNotIn(Assignment([a_, b , c_]), assignments)
        self.assertNotIn(Assignment([a_, b_, c ]), assignments)
        self.assertNotIn(Assignment([a_, b_, c_]), assignments)
        assignments = Assignment.generate([A, B, C])
        self.assertIn(Assignment([a , b , c ]), assignments)
        self.assertIn(Assignment([a , b , c_]), assignments)
        self.assertIn(Assignment([a , b_, c ]), assignments)
        self.assertIn(Assignment([a , b_, c_]), assignments)
        self.assertIn(Assignment([a_, b , c ]), assignments)
        self.assertIn(Assignment([a_, b , c_]), assignments)
        self.assertIn(Assignment([a_, b_, c ]), assignments)
        self.assertIn(Assignment([a_, b_, c_]), assignments)

class TestQuery(unittest.TestCase):
    def test_query(self):
        A, B = map(Variable, 'AB')
        a, a_ = A
        b, b_ = B
        query = Query.from_natural(A, B)
        self.assertListEqual(query.query, [A, B])
        self.assertListEqual(query.query_vars, [A, B])
        self.assertListEqual(query.given, [])
        self.assertListEqual(query.given_vars, [])
        self.assertTrue(query.is_marginal_query)
        self.assertFalse(query.is_conditional_query)
        query = Query.from_natural(a, b)
        self.assertListEqual(query.query, [a, b])
        self.assertListEqual(query.query_vars, [A, B])
        self.assertListEqual(query.given, [])
        self.assertListEqual(query.given_vars, [])
        self.assertFalse(query.is_marginal_query)
        self.assertFalse(query.is_conditional_query)
        query = Query.from_natural(a|B)
        self.assertListEqual(query.query, [a])
        self.assertListEqual(query.query_vars, [A])
        self.assertListEqual(query.given, [B])
        self.assertListEqual(query.given_vars, [B])
        self.assertFalse(query.is_marginal_query)
        self.assertTrue(query.is_conditional_query)
        self.assertTrue(query.is_full_conditional_query)
        query = Query.from_natural(A|b)
        self.assertListEqual(query.query, [A])
        self.assertListEqual(query.query_vars, [A])
        self.assertListEqual(query.given, [b])
        self.assertListEqual(query.given_vars, [B])
        self.assertTrue(query.is_marginal_query)
        self.assertTrue(query.is_conditional_query)
        self.assertFalse(query.is_full_conditional_query)
        query = Query.from_natural(A|B)
        self.assertListEqual(query.query, [A])
        self.assertListEqual(query.query_vars, [A])
        self.assertListEqual(query.given, [B])
        self.assertListEqual(query.given_vars, [B])
        self.assertTrue(query.is_marginal_query)
        self.assertTrue(query.is_conditional_query)
        self.assertTrue(query.is_full_conditional_query)
        self.assertTupleEqual(A | B, (A, B))
        self.assertTupleEqual(a | B, (a, B))
        self.assertTupleEqual(A | b, (A, b))
        self.assertTupleEqual(a | b, (a, b))

class TestNumberSystems(unittest.TestCase):
    def test_float(self):
        pass
        # TODO: Implement test cases.
    def test_decimal(self):
        pass
        # TODO: Implement test cases.
    def test_sympy_integer(self):
        pass
        # TODO: Implement test cases.
    def test_sympy_float(self):
        pass
        # TODO: Implement test cases.

class TestTables(unittest.TestCase):
    def test_marginal_table(self):
        A, B = map(Variable, 'AB')
        a, a_ = A
        b, b_ = B
        P = Table([A, B])
        self.assertFalse(P.is_valid)
        P[a , b ] = 0.3
        P[a , b_] = 0.2
        P[a_, b ] = 0.4
        P[a_, b_] = 0.09
        self.assertFalse(P.is_valid)
        self.assertAlmostEqual(P.total_probability(), 0.99)
        with self.assertRaises(AssertionError):
            P.marginalize_over([A])
        with self.assertRaises(AssertionError):
            P.marginalize_out([A])
        with self.assertRaises(AssertionError):
            P.condition([A], [B])
        with self.assertRaises(AssertionError):
            P.condition_on([A])
        with self.assertRaises(AssertionError):
            P.get_samples()
        with self.assertRaises(AssertionError):
            P(A)
        P[a_, b_] = 0.1
        self.assertTrue(P.is_valid)
        self.assertAlmostEqual(P.total_probability(), 1.0)
        self.assertEqual(P.marginalize_over([A]).variables, frozenset([A]))
        self.assertEqual(P.marginalize_out([A]).variables, frozenset([B]))
        self.assertEqual(P.condition([A], [B]).context_variables, frozenset([B]))
        self.assertEqual(P.condition_on([A]).context_variables, frozenset([A]))
        self.assertEqual(P.get_samples(header=[A, B])[0], [A, B])
        self.assertEqual(P(A).variables, frozenset([A]))
        self.assertAlmostEqual(P(a), 0.5)
        self.assertAlmostEqual(P(a, b_), 0.2)
        self.assertAlmostEqual(P.query(Query([a_, b_])), 0.1)
        self.assertEqual(P(A|B).context_variables, frozenset([B]))
        self.assertAlmostEqual(P(b|a), 0.6)
        P.randomize()
        self.assertTrue(P.is_valid)
        P[a_, b_] += 0.1
        self.assertFalse(P.is_valid)
        P.normalize()
        self.assertTrue(P.is_valid)
    def test_marginal_table_multiplication(self):
        C = Variable('C', ['Heads', 'Tails'], description='Coin')
        D = Variable('D', [1, 2, 3, 4, 5, 6], description='Die')
        h, t = C
        d1, d2, d3, d4, d5, d6 = D
        Pc = Table([C])
        Pc[h] = Pc[t] = 0.5
        Pd = Table([D])
        Pd[d1] = Pd[d2] = Pd[d3] = Pd[d4] = Pd[d5] = Pd[d6] = 1.0/6.0
        with self.assertRaises(ValueError):
            Pc * Pc
        P = Pc * Pd
        for assignment in Assignment.generate([C, D]):
            self.assertAlmostEqual(P[assignment], 1.0/12.0)
    def test_marginal_table_learning_from_complete_data(self):
        '''
        See Figure 17.2 in Modeling and Reasoning with Bayesian Networks (Darwiche).
        '''
        S, H, E = map(Variable, 'SHE')
        h, h_ = H
        s, s_ = S
        e, e_ = E
        data_header = [H, S, E]
        data_samples = [
            (True, False, True),
            (True, False, True),
            (False, True, False),
            (False, False, True),
            (True, False, False),
            (True, False, True),
            (False, False, False),
            (True, False, True),
            (True, False, True),
            (False, False, True),
            (True, False, True),
            (True, True, True),
            (True, False, True),
            (True, True, True),
            (True, False, True),
            (True, False, True)]
        data_assignments = data_to_assignments(data_header, data_samples)
        P = Table([S, H, E])
        self.assertFalse(P.is_valid)
        P.learn_from_complete_data(data_assignments)
        self.assertAlmostEqual(P(h , s , e ), 2.0/16.0)
        self.assertAlmostEqual(P(h , s , e_), 0.0/16.0)
        self.assertAlmostEqual(P(h , s_, e ), 9.0/16.0)
        self.assertAlmostEqual(P(h , s_, e_), 1.0/16.0)
        self.assertAlmostEqual(P(h_, s , e ), 0.0/16.0)
        self.assertAlmostEqual(P(h_, s , e_), 1.0/16.0)
        self.assertAlmostEqual(P(h_, s_, e ), 2.0/16.0)
        self.assertAlmostEqual(P(h_, s_, e_), 1.0/16.0)
        self.assertAlmostEqual(P.data_likelihood(data_assignments[0:1]), 9.0/16.0)
        self.assertAlmostEqual(P.data_likelihood(data_assignments[0:2]), (9.0/16.0)**2.0)
        self.assertAlmostEqual(P.data_log_likelihood(data_assignments[0:1]), log(9.0/16.0))
        self.assertAlmostEqual(P.data_log_likelihood(data_assignments[0:2]), log((9.0/16.0)**2.0))
        self.assertAlmostEqual(P.data_likelihood(data_assignments), 3.36033708671e-10)
        self.assertAlmostEqual(P.data_log_likelihood(data_assignments), -21.8138096376)
    def test_marginal_table_expectation_maximization(self):
        A, B, C = map(Variable, 'ABC')
        a, a_ = A
        b, b_ = B
        c, c_ = C
        Pi = Table([A, B, C])
        Pi[a , b , c ] = 0.125
        Pi[a , b , c_] = 0.125
        Pi[a , b_, c ] = 0.125
        Pi[a , b_, c_] = 0.125
        Pi[a_, b , c ] = 0.125
        Pi[a_, b , c_] = 0.125
        Pi[a_, b_, c ] = 0.125
        Pi[a_, b_, c_] = 0.125
        data_header = [A, B, C]
        data_samples = [
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, None, None),
            (True, None, None),
            (True, None, None),
            (True, None, None),]
        data_assignments = data_to_assignments(data_header, data_samples)
        P = Table([A, B, C])
        P.learn_with_expectation_maximization(data_assignments, Pi)
        self.assertAlmostEqual(P[a , b , c ], 0.125)
        self.assertAlmostEqual(P[a , b , c_], 0.125)
        self.assertAlmostEqual(P[a , b_, c ], 0.125)
        self.assertAlmostEqual(P[a , b_, c_], 0.125)
        self.assertAlmostEqual(P[a_, b , c ], 0.125)
        self.assertAlmostEqual(P[a_, b , c_], 0.125)
        self.assertAlmostEqual(P[a_, b_, c ], 0.125)
        self.assertAlmostEqual(P[a_, b_, c_], 0.125)
        data_samples = [
            (None, False, False),
            (False, False, None),
            (False, None, False),
            (False, None, True),
            (True, False, None),
            (None, False, True),
            (True, True, None),
            (True, None, True),]
        data_assignments = data_to_assignments(data_header, data_samples)
        pre_learning_log_likelihood = Pi.data_log_likelihood(data_assignments)
        P.learn_with_expectation_maximization(data_assignments, Pi)
        post_learning_log_likelihood = P.data_log_likelihood(data_assignments)
        self.assertGreater(post_learning_log_likelihood, pre_learning_log_likelihood)
    def test_conditional_table(self):
        A, B, C = map(Variable, 'ABC')
        a, a_ = A
        b, b_ = B
        c, c_ = C
        P = Table([C], [A, B])
        P[a , b , c ] = 1
        P[a , b , c_] = 0
        P[a , b_, c ] = 0
        P[a , b_, c_] = 1
        P[a_, b , c ] = 0
        P[a_, b , c_] = 1
        P[a_, b_, c ] = 0
        P[a_, b_, c_] = 0.9
        self.assertFalse(P.is_valid)
        P[a_, b_, c_] = 1
        self.assertTrue(P.is_valid)
    #def test_conditional_table_multiplication(self):
    #    A, B, C = map(Variable, 'ABC')
    #    a, a_ = A
    #    b, b_ = B
    #    c, c_ = C
    #    Pc = Table([C], [A, B])
    #    Pc[a , b , c ] = 0.2
    #    Pc[a , b , c_] = 0.8
    #    Pc[a , b_, c ] = 0.3
    #    Pc[a , b_, c_] = 0.7
    #    Pc[a_, b , c ] = 0.4
    #    Pc[a_, b , c_] = 0.6
    #    Pc[a_, b_, c ] = 0.5
    #    Pc[a_, b_, c_] = 0.5
    #    Pj = Table([A, B])
    #    Pj[a , b ] = 0.1
    #    Pj[a , b_] = 0.2
    #    Pj[a_, b ] = 0.3
    #    Pj[a_, b_] = 0.4
    #    P = Pc * Pj
    #    self.assertTrue(P.is_conditional_table)
    #    self.assertAlmostEqual(P[a , b , c ], Pc[a , b , c ] * Pj[a , b ])
    #    self.assertAlmostEqual(P[a , b , c_], Pc[a , b , c_] * Pj[a , b ])
    #    self.assertAlmostEqual(P[a , b_, c ], Pc[a , b_, c ] * Pj[a , b_])
    #    self.assertAlmostEqual(P[a , b_, c_], Pc[a , b_, c_] * Pj[a , b_])
    #    self.assertAlmostEqual(P[a_, b , c ], Pc[a_, b , c ] * Pj[a_, b ])
    #    self.assertAlmostEqual(P[a_, b , c_], Pc[a_, b , c_] * Pj[a_, b ])
    #    self.assertAlmostEqual(P[a_, b_, c ], Pc[a_, b_, c ] * Pj[a_, b_])
    #    self.assertAlmostEqual(P[a_, b_, c_], Pc[a_, b_, c_] * Pj[a_, b_])
    #    P = P(C)
    #    self.assertAlmostEqual(P[c], Pc[a, b, c]*Pj[a, b] + Pc[a, b_, c]*Pj[a, b_] + Pc[a_, b, c]*Pj[a_, b] + Pc[a_, b_, c]*Pj[a_, b_])
    #    self.assertAlmostEqual(P[c_], Pc[a, b, c_]*Pj[a, b] + Pc[a, b_, c_]*Pj[a, b_] + Pc[a_, b, c_]*Pj[a_, b] + Pc[a_, b_, c_]*Pj[a_, b_])
    #    Pj = Table([A])
    #    Pj[a ] = 0.4
    #    Pj[a_] = 0.6
    #    P = Pc * Pj
    #    self.assertTrue(P.is_conditional_table)
    #    self.assertAlmostEqual(P[a , b , c ], Pc[a , b , c ] * Pj[a ])
    #    self.assertAlmostEqual(P[a , b , c_], Pc[a , b , c_] * Pj[a ])
    #    self.assertAlmostEqual(P[a , b_, c ], Pc[a , b_, c ] * Pj[a ])
    #    self.assertAlmostEqual(P[a , b_, c_], Pc[a , b_, c_] * Pj[a ])
    #    self.assertAlmostEqual(P[a_, b , c ], Pc[a_, b , c ] * Pj[a_])
    #    self.assertAlmostEqual(P[a_, b , c_], Pc[a_, b , c_] * Pj[a_])
    #    self.assertAlmostEqual(P[a_, b_, c ], Pc[a_, b_, c ] * Pj[a_])
    #    self.assertAlmostEqual(P[a_, b_, c_], Pc[a_, b_, c_] * Pj[a_])

class TestGraph(unittest.TestCase):
    def test_dag(self):
        pass
        # TODO: Implement test cases.

class TestBayesianNetwork(unittest.TestCase):
    def test_basics(self):
        pass
        # TODO: Implement test cases.
    def test_deterministic_network(self):
        pass
        # TODO: Implement test cases.

class TestInformationTheory(unittest.TestCase):
    def test_entropy(self):
        pass
        # TODO: Implement test cases.

if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_loader.loadTestsFromTestCase(TestBasics).debug()
    test_loader.loadTestsFromTestCase(TestQuery).debug()
    test_loader.loadTestsFromTestCase(TestTables).debug()
    test_loader.loadTestsFromTestCase(TestGraph).debug()
    test_loader.loadTestsFromTestCase(TestBayesianNetwork).debug()
    test_loader.loadTestsFromTestCase(TestInformationTheory).debug()
    unittest.main()

