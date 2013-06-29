import collections
import math
import random
import sys

from discrete_probability import *

if __name__ == '__main__':
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
