import collections
import math
import numpy as np

class Variable():
	def __init__(self, name, description='', values=(True, False)):
		self.name = name
		self.description = name if description == '' else description
		self.values = values
	def __str__(self):
		return self.name
	def __repr__(self):
		return str(self)
	def __lt__(self, other):
		if isinstance(other, Variable):
			return DirectedEdge(other, self, False)
		raise ValueError('Expecting Variable.')
	def __gt__(self, other):
		if isinstance(other, Variable):
			return DirectedEdge(self, other, True)
		raise ValueError('Expecting Variable.')
	def __or__(self, other):
		return (self, other)
	def __lshift__(self, other):
		return SingleAssignment(self, other)

# TODO: This should probably be a derivative of a tuple. (Viet Nguyen, 2013-06-23)
BaseAssignment = collections.namedtuple('BaseAssignment', ['variable', 'value'])
class SingleAssignment(BaseAssignment):
	def __str__(self):
		return '{!s}={!s}'.format(self.variable, self.value)
	def __repr__(self):
		return str(self)

class Assignment(frozenset):
	def __str__(self):
		return '({:})'.format(', '.join([str(x) for x in self]))
	def __repr__(self):
		return str(self)
	def get_variables(self):
		return frozenset([x.variable for x in self])
	@staticmethod
	def generate(variables, trace=[]):
		if len(variables) == 0:
			return [Assignment(trace)]
		else:
			variable, rest = variables[0], variables[1:]
			traces = []
			for value in variable.values:
				traces.extend(Assignment.generate(rest, trace+[SingleAssignment(variable, value)]))
			return traces
	@staticmethod
	def complete(variables, partial_assignment):
		return Assignment.generate(list(set(variables).difference(partial_assignment.get_variables())), list(partial_assignment))

class JointTable():
	def __init__(self, variables):
		self.variables = set(variables)
		self.assignments = Assignment.generate(list(self.variables))
		self.probabilities = {}
		for assignment in self.assignments:
			self.probabilities[assignment] = None
	def set_row(self, assignment, value):
		self.probabilities[assignment] = value
	def learn_from_complete_data(self, header, data):
		total_count = float(len(data))
		for assignment in self.assignments:
			self.probabilities[assignment] = 0.0
		for sample in data:
			assignment = Assignment([SingleAssignment(variable, value) for variable, value in zip(header, sample)])
			self.probabilities[assignment] += 1.0
		for assignment in self.assignments:
			self.probabilities[assignment] /= total_count
	def validate(self):
		if None in self.probabilities.values():
			return False
		# TODO: This is probably too stringent.
		if sum(self.probabilities.values()) != 1.0:
			return False
		return True
	def __call__(self, *args):
		args = list(args)
		query_vars = []
		given_vars = []
		separator_index = filter(lambda x: not isinstance(x[1], Variable), enumerate(args))
		if separator_index == []:
			query_vars = args
		else:
			separator_index = separator_index[0][0]
			query_vars = args[0:separator_index] + [args[separator_index][0]]
			given_vars = [args[separator_index][1]] + args[separator_index+1:]
		print('{:} given {:}'.format(str(query_vars), str(given_vars)))

class ConditionalTable():
	pass

class DirectedEdge():
	def __init__(self, from_var, to_var, right=True):
		self.from_var = from_var
		self.to_var = to_var
		self.right = right
	def __str__(self):
		return '{:} > {:}'.format(str(self.from_var), str(self.to_var))
	def __repr__(self):
		return str(self)

class Network():
	def __init__(self, variables, edges):
		self.variables = set(variables)
		self.edges = edges
		#self.check_for_cycles()
		self.gather_families()
	def check_for_cycles(self):
		raise NotImplementedError
	def gather_families(self):
		self.root_variables = set()
		self.families = {}
		for variable in self.variables:
			parents = []
			for edge in self.edges:
				if edge.to_var == variable:
					parents.append(edge.from_var)
			self.families[variable] = parents
			if len(parents) == 0:
				self.root_variables.add(variable)
		self.parental_variables = set()
		for edge in self.edges:
			self.parental_variables.add(edge.from_var)
		self.leaf_variables = set(self.variables).difference(self.parental_variables)
		self.parameterization = {}
		for variable in self.variables:
			self.parameterization[variable] = None
	def as_joint_table(self):
		joint_table = JointTable(self.variables)
		return joint_table

S, H, E = map(Variable, ['S', 'H', 'E'])
network = Network([S, H, E], [S < H, H > E])
print(network.variables)
print(network.edges)
P = network.as_joint_table()
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
P.learn_from_complete_data([H, S, E], data)
print(P.assignments)
print(P.validate())
print([P.probabilities[assignment] for assignment in P.assignments])
print(Assignment.complete(network.variables, Assignment([S<<True])))