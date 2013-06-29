from random import randint

from .. import JointTable, ConditionalTable

class DirectedEdge():
	def __init__(self, from_var, to_var, right=True):
		self.from_var = from_var
		self.to_var = to_var
		self.right = right
	def __str__(self):
		return '{:} > {:}'.format(str(self.from_var), str(self.to_var))
	def __repr__(self):
		return str(self)

#def gather_families(self):
#	self.root_variables = set()
#	self.families = {}
#	for variable in self.variables:
#		parents = []
#		for edge in self.edges:
#			if edge.to_var == variable:
#				parents.append(edge.from_var)
#		self.families[variable] = set(parents)
#		if len(parents) == 0:
#			self.root_variables.add(variable)
#	self.parental_variables = set()
#	for edge in self.edges:
#		self.parental_variables.add(edge.from_var)
#	self.leaf_variables = set(self.variables).difference(self.parental_variables)
#	self.conditionals = {}
#	for variable in self.variables:
#		self.conditionals[variable] = ConditionalTable([variable], self.families[variable])

class BayesianNetwork():
	def __init__(self, variables, edges):
		self.variables = frozenset(variables)
		self.edges = frozenset(edges)
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
			self.families[variable] = set(parents)
			if len(parents) == 0:
				self.root_variables.add(variable)
		self.parental_variables = set()
		for edge in self.edges:
			self.parental_variables.add(edge.from_var)
		self.leaf_variables = set(self.variables).difference(self.parental_variables)
		self.conditionals = {}
		for variable in self.variables:
			self.conditionals[variable] = ConditionalTable([variable], self.families[variable])
	def validate(self):
		for variable in self.variables:
			if not self.conditionals[variable].is_valid:
				return False
		return True
	is_valid = property(validate)
	def randomize(self):
		for conditional in self.conditionals.values():
			conditional.randomize()
		return self
	def learn_from_complete_data(self, header, data):
		total_count = float(len(data))
		accum_assignments = set()
		for variable in self.variables:
			conditional = self.conditionals[variable]
			accum_assignments.update(conditional.all_assignments)
			accum_assignments.update(conditional.context_assignments)
			for context_assignment in conditional.context_assignments:
				context_table = conditional.context_tables[context_assignment]
				for assignment in conditional.assignments:
					context_table.probabilities[assignment] = 0.0
		accumulators = {}
		for accum_assignment in accum_assignments:
