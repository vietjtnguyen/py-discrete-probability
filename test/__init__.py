import unittest

from discrete_probability import *

class TestBasics(unittest.TestCase):
	def test_helpers(self):
		self.assertIn(weighted_choose([(0.1, 'A'), (0.4, 'B'), (0.5, 'C')]), ['A', 'B', 'C'])
		with self.assertRaises(ValueError):
			for i in range(0, 100000):
				weighted_choose([(0.5, 'A'), (0.4, 'B')])
			self.fail("You're very unlucky.")
		A, B = map(Variable, 'AB')
		a, a_ = A.get_assignments()
		b, b_ = B.get_assignments()
		query, query_vars, given, given_vars, is_marginal_query, is_conditional_query, is_full_conditional_query = parse_query(A, B)
		self.assertListEqual(query, [A, B])
		self.assertListEqual(query_vars, [A, B])
		self.assertListEqual(given, [])
		self.assertListEqual(given_vars, [])
		self.assertTrue(is_marginal_query)
		self.assertFalse(is_conditional_query)
		query, query_vars, given, given_vars, is_marginal_query, is_conditional_query, is_full_conditional_query = parse_query(a, b)
		self.assertListEqual(query, [a, b])
		self.assertListEqual(query_vars, [A, B])
		self.assertListEqual(given, [])
		self.assertListEqual(given_vars, [])
		self.assertFalse(is_marginal_query)
		self.assertFalse(is_conditional_query)
		query, query_vars, given, given_vars, is_marginal_query, is_conditional_query, is_full_conditional_query = parse_query(a|B)
		self.assertListEqual(query, [a])
		self.assertListEqual(query_vars, [A])
		self.assertListEqual(given, [B])
		self.assertListEqual(given_vars, [B])
		self.assertFalse(is_marginal_query)
		self.assertTrue(is_conditional_query)
		self.assertTrue(is_full_conditional_query)
		query, query_vars, given, given_vars, is_marginal_query, is_conditional_query, is_full_conditional_query = parse_query(A|b)
		self.assertListEqual(query, [A])
		self.assertListEqual(query_vars, [A])
		self.assertListEqual(given, [b])
		self.assertListEqual(given_vars, [B])
		self.assertTrue(is_marginal_query)
		self.assertTrue(is_conditional_query)
		self.assertFalse(is_full_conditional_query)
		query, query_vars, given, given_vars, is_marginal_query, is_conditional_query, is_full_conditional_query = parse_query(A|B)
		self.assertListEqual(query, [A])
		self.assertListEqual(query_vars, [A])
		self.assertListEqual(given, [B])
		self.assertListEqual(given_vars, [B])
		self.assertTrue(is_marginal_query)
		self.assertTrue(is_conditional_query)
		self.assertTrue(is_full_conditional_query)
	def test_binary_variable(self):
		A = Variable('A')
		self.assertEqual(A.values, (True, False))
		self.assertEqual(A.description, 'A')
		self.assertEqual(SingleAssignment(A, True), A << True)
		self.assertEqual(SingleAssignment(A, False), A << False)
		self.assertEqual(A.get_assignments(), (A << True, A << False))
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
		self.assertEqual(B.get_assignments(), (B << 0, B << 1, B << 'x', B << 'y'))
		self.assertTrue(isinstance(B << 0, SingleAssignment))
		self.assertTrue(isinstance(B << 1, SingleAssignment))
		self.assertTrue(isinstance(B << 'x', SingleAssignment))
		self.assertTrue(isinstance(B << 'y', SingleAssignment))
		with self.assertRaises(ValueError):
			B << 'a'
		# TODO: Test __or__
	def test_assignment(self):
		pass
		# TODO: Test consistent_with
		# TODO: Test project
		# TODO: Test get_variable
		# TODO: Test get_variables
		# TODO: Test ordered
		# TODO: Test ordered_values
		# TODO: Test complete
		# TODO: Test generate

class TestTables(unittest.TestCase):
	def test_joint_table(self):
		A, B = map(Variable, 'AB')
		a, a_ = A.get_assignments()
		b, b_ = B.get_assignments()
		P = JointTable([A, B])
		self.assertTrue(not P.is_valid)
		P[a , b ] = 0.3
		P[a , b_] = 0.2
		P[a_, b ] = 0.4
		P[a_, b_] = 0.09
		self.assertTrue(not P.is_valid)
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
		self.assertAlmostEqual(P.query(a_, b_), 0.1)
		self.assertEqual(P(A|B).context_variables, frozenset([B]))
		self.assertAlmostEqual(P(b|a), 0.6)
		P.randomize()
		self.assertTrue(P.is_valid)
		P[a_, b_] += 0.1
		self.assertTrue(not P.is_valid)
		P.normalize()
		self.assertTrue(P.is_valid)
	def test_joint_table_learning_from_complete_data(self):
		'''
		See Figure 17.2 in Modeling and Reasoning with Bayesian Networks (Darwiche).
		'''
		S, H, E = map(Variable, 'SHE')
		h, h_ = H.get_assignments()
		s, s_ = S.get_assignments()
		e, e_ = E.get_assignments()
		header=[H, S, E]
		data = [
			[True, False, True],
			[True, False, True],
			[False, True, False],
			[False, False, True],
			[True, False, False],
			[True, False, True],
			[False, False, False],
			[True, False, True],
			[True, False, True],
			[False, False, True],
			[True, False, True],
			[True, True, True],
			[True, False, True],
			[True, True, True],
			[True, False, True],
			[True, False, True]]
		P = JointTable([S, H, E])
		self.assertTrue(not P.is_valid)
		P.learn_from_complete_data(header, data)
		self.assertEqual(P(h , s , e ), 2.0/16.0)
		self.assertEqual(P(h , s , e_), 0.0/16.0)
		self.assertEqual(P(h , s_, e ), 9.0/16.0)
		self.assertEqual(P(h , s_, e_), 1.0/16.0)
		self.assertEqual(P(h_, s , e ), 0.0/16.0)
		self.assertEqual(P(h_, s , e_), 1.0/16.0)
		self.assertEqual(P(h_, s_, e ), 2.0/16.0)
		self.assertEqual(P(h_, s_, e_), 1.0/16.0)
	def test_conditional_table(self):
		pass

class TestGraph(unittest.TestCase):
	def test_dag(self):
		pass
		# TODO: Implement test cases.

class TestBayesianNetwork(unittest.TestCase):
	def test_basics(self):
		pass
		# TODO: Implement test cases.

class TestInformationTheory(unittest.TestCase):
	def test_entropy(self):
		pass
		# TODO: Implement test cases.

if __name__ == '__main__':
	unittest.main()
	print('')
	print(Assignment([s, h, e_]))
	print(Assignment([s, h, e_]).project([S, H]))
	print(Assignment([s, h, e_]).project([E]))
	print(P.assignments)
	print(len(P.assignments))
	print(P.validate())
	print([P.probabilities[assignment] for assignment in P.assignments])
	print(P)
	print('')
	print(P.marginalize_out([S]))
	print('')
	print(P.marginalize_out([H]))
	print('')
	print(P.marginalize_out([E]))
	print('')
	print(P.marginalize_out([E,S]))
	print('')
	print(P.marginalize_over([E,S]))
	print('')
	print(sum([Assignment([S<<True]).consistent_with(Assignment([SingleAssignment(variable, value) for variable, value in zip(header, sample)])) for sample in data]))
	print(sum([Assignment([S<<True, H<<True]).consistent_with(Assignment([SingleAssignment(variable, value) for variable, value in zip(header, sample)])) for sample in data]))
	print(sum([Assignment([H<<True]).consistent_with(Assignment([SingleAssignment(variable, value) for variable, value in zip(header, sample)])) for sample in data]))
	print('')
	P_H = ConditionalTable([S], [H])
	print(P_H)
	print(P_H.is_valid)
	print('')
	print(P.condition_on([H]))
	print(P.condition_on([H]).is_valid)
	print('')
	print(P.condition([S], [H]))
	print(P.condition([S], [H]).is_valid)
	print('')
	#P = JointTable([S, H, E])
	#P.randomize()
	print(P(e))
	print(P(H,S|e))
	print(P(h,s|e))
	print(P(H|e))
	print(P(h|e))

	network = BayesianNetwork([S, H, E], [S < H, H > E])
	network.learn_from_complete_data(header, data)
	print('')
	print(network.variables)
	print(network.edges)
	for variable in network.variables:
		print(network.conditionals[variable])
	P_b = network.as_joint_table()
	print(P_b)
	print(P)
	print(P.get_samples())
	print(P.get_samples(100))
	print(P.get_samples(100, [S,H,E]))
	print(network.simulate())
	print(JointTable([S,H,E]).learn_from_complete_data(*P.get_samples(1000)))
	print(JointTable([S,H,E]).learn_from_complete_data(*P.get_samples(1000, [S,H,E])))
	A,B,C,D,E,F,G,H = map(Variable, 'ABCDEFGH')
	nb = BayesianNetwork([A,B,C,D,E,F,G,H], [A>B,B>C,C>D,D>E,E>F,F>G,G>H])
	print(nb)
	print(nb.topological_order)
	nb = BayesianNetwork([A,B,C,D], [A>C,B>C,A>B,D>B]).randomize()
	Pb = nb.as_joint_table()
	print(nb)
	print(Pb)
	print(Pb.is_valid)
	print(Pb(A))
	print(Pb(F))
	print(Pb(A,F))
	print(nb.topological_order)
	print(nb.simulate())

