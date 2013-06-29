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

