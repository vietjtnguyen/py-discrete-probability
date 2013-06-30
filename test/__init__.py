import unittest

from discrete_probability import *

class TestBasics(unittest.TestCase):
	def test_single_assignment(self):
		A = Variable('A')
		self.assertEqual(A.values, (True, False))
		self.assertEqual(A.description, 'A')
		self.assertEqual(SingleAssignment(A, True), A << True)
		self.assertEqual(SingleAssignment(A, False), A << False)
		self.assertEqual(A.get_assignments(), (A << True, A << False))

		B = Variable('B', (0, 1, 'x', 'y'), 'multiple values')
		self.assertEqual(B.values, (0, 1, 'x', 'y'))
		self.assertEqual(B.description, 'multiple values')
		self.assertEqual(SingleAssignment(B, 0), B << 0)
		self.assertEqual(SingleAssignment(B, 1), B << 1)
		self.assertEqual(SingleAssignment(B, 'x'), B << 'x')
		self.assertEqual(SingleAssignment(B, 'y'), B << 'y')
		self.assertEqual(B.get_assignments(), (B << 0, B << 1, B << 'x', B << 'y'))
		
		A << 0
		A << 1
		with self.assertRaises(ValueError):
			A << 2
		B << 0
		B << 1
		with self.assertRaises(ValueError):
			B << 'a'
	def test_joint_table(self):
		pass

if __name__ == '__main__':
	unittest.main()
	S, H, E = variables = map(Variable, ['S', 'H', 'E'])
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
	print('')
	print(Assignment([s, h, e_]))
	print(Assignment([s, h, e_]).project([S, H]))
	print(Assignment([s, h, e_]).project([E]))
	P = JointTable(variables)
	P.learn_from_complete_data(header, data)
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

