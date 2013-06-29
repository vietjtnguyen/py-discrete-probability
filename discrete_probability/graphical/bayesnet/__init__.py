from random import randint

from .. import *

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
			accumulators[accum_assignment] = 0
		for accum_assignment in accum_assignments:
			for sample in data:
				sample_assignment = Assignment([SingleAssignment(variable, value) for variable, value in zip(header, sample)])
				if accum_assignment.consistent_with(sample_assignment):
					accumulators[accum_assignment] += 1
		for variable in self.variables:
			conditional = self.conditionals[variable]
			for context_assignment in conditional.context_assignments:
				context_table = conditional.context_tables[context_assignment]
				for assignment in conditional.assignments:
					context_table.probabilities[assignment] = float(accumulators[assignment.union(context_assignment)])/float(accumulators[context_assignment])
		return self
	def direct_sample(self):
		raise NotImplementedError
	def as_joint_table(self):
		joint_table = JointTable(self.variables)
		for assignment in joint_table.assignments:
			product = 1.0
			for variable in self.variables:
				conditional = self.conditionals[variable]
				product *= conditional.context_tables[assignment.project(self.families[variable])].probabilities[assignment.project([variable])]
			joint_table.probabilities[assignment] = product
		return joint_table
	def get_display_js(self, width=640, height=480):
		with open('graph_display.js', 'r') as f:
			return 'var links=[{:}];var w={:},h={:};{:}'.format(''.join(['{{source:"{:}",target:"{:}"}},'.format(edge.from_var, edge.to_var) for edge in self.edges]), width, height, f.read())
	def display(self, width=640, height=480):
		import IPython.display
		div_id = 'probgraphdisplay'+str(randint(0, 65536))
		IPython.display.display(IPython.display.HTML(data='<div id="{:}"></div>'.format(div_id)))
		IPython.display.display(IPython.display.Javascript(data='var cellDivId="{:}";{:}'.format(div_id, self.get_display_js(width, height)), lib='http://d3js.org/d3.v3.min.js', css='/files/graph_display.css'))
		return self

