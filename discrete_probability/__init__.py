__version__ = '0.8dev'

from collections import namedtuple
from math import log
from random import random, randint
from sys import float_info

################################################################################
# Utilities
################################################################################

def weighted_choose(weighted_choices):
	x = random()
	for weight, choice in weighted_choices:
		if x <= weight:
			return choice
		x -= weight
	raise ValueError('Total probability of choices does not sum to one.')

def parse_query(*args):
	args = list(args)
	query = []
	given = []
	separator_index = filter(lambda x: not (isinstance(x[1], SingleAssignment) or isinstance(x[1], Variable)), enumerate(args))
	if separator_index == []:
		query = args
	else:
		separator_index = separator_index[0][0]
		query = args[0:separator_index] + [args[separator_index][0]]
		given = [args[separator_index][1]] + args[separator_index+1:]
	query_vars = map(lambda x: x if isinstance(x, Variable) else x.variable, query)
	given_vars = map(lambda x: x if isinstance(x, Variable) else x.variable, given)
	is_marginal_query = len(filter(lambda x: isinstance(x, Variable), query)) > 0
	is_conditional_query = len(given) > 0
	is_full_conditional_query = len(filter(lambda x: isinstance(x, Variable), given)) > 0
	return query, query_vars, given, given_vars, is_marginal_query, is_conditional_query, is_full_conditional_query

################################################################################
# Discrete random variables
################################################################################

class Variable():
	def __init__(self, name, values=(True, False), description=''):
		self.name = name
		self.description = name if description == '' else description
		self.values = values
		self.assignments = tuple(self<<value for value in self.values)
		if None in self.values:
			raise ValueError('Cannot use None as a value. None is reserved for missing data.')
	def __len__(self):
		return len(self.values)
	def __getitem__(self, key):
		return self.assignments[key]
	def __iter__(self):
		return iter(self.assignments)
	def __str__(self):
		return self.name
	def __repr__(self):
		return str(self)
	def column_width(self):
		return max(len(str(self)), max(*[len(str(value)) for value in self.values]))
	def __lt__(self, other):
		if isinstance(other, Variable):
			return DirectedEdge(other, self)
		raise ValueError('Expecting Variable.')
	def __gt__(self, other):
		if isinstance(other, Variable):
			return DirectedEdge(self, other)
		raise ValueError('Expecting Variable.')
	def __or__(self, other):
		return (self, other)
	def __lshift__(self, other):
		if other not in self.values:
			raise ValueError('Assigned value is not valid for this variable: {:} not in {:}.'.format(other, self.values))
		return SingleAssignment(self, other)

BaseAssignment = namedtuple('BaseAssignment', ['variable', 'value'])
class SingleAssignment(BaseAssignment):
	def __init__(self, variable, value):
		super(SingleAssignment, self).__init__(variable, value)
		if not value in variable.values:
			raise ValueError('Assigned incompatible value to variable. Value {1} not in {2} for variable {0}.'.format(variable, value, str(variable.values)))
	def __str__(self):
		return '{!s}={!s}'.format(self.variable, self.value)
	def __repr__(self):
		return str(self)
	def __or__(self, other):
		return (self, other)

class Assignment(frozenset):
	def __new__(_cls, single_assignments):
		if isinstance(single_assignments, SingleAssignment):
			return frozenset.__new__(_cls, [single_assignments])
		else:
			if len(filter(lambda x: not isinstance(x, SingleAssignment), single_assignments)) > 0:
				raise ValueError('Assignments can only be made from SingleAssignments.')
			return frozenset.__new__(_cls, single_assignments)
	def __str__(self):
		return '({:})'.format(', '.join([str(x) for x in self]))
	def __repr__(self):
		return str(self)
	def consistent_with(self, other):
		return len(self.union(other)) == len(self.union(other).get_variables())
	def project(self, variables):
		return Assignment(filter(lambda x: x.variable in variables, self))
	def get_variable(self, variable):
		return filter(lambda x: x.variable == variable, self)[0]
	def get_variables(self):
		return frozenset([x.variable for x in self])
	def ordered(self, order):
		return tuple(self.get_variable(variable) for variable in order)
	def ordered_values(self, order):
		return tuple(self.get_variable(variable).value for variable in order)
	def complete(self, variables):
		return Assignment.generate(set(variables).difference(self.get_variables()), list(self))
	@staticmethod
	def generate(variables, trace=[]):
		variables = list(variables)
		if len(variables) == 0:
			return [Assignment(trace)]
		else:
			variable, rest = variables[0], variables[1:]
			traces = []
			for value in variable.values:
				traces.extend(Assignment.generate(rest, trace+[SingleAssignment(variable, value)]))
			return traces
Assignment.empty = Assignment(())

################################################################################
# Discrete probability tables
################################################################################

class MarginalTable():
	def __init__(self, variables, context_assigment=()):
		self.variables = frozenset(variables)
		self.context_assignment = context_assigment
		self.assignments = Assignment.generate(self.variables)
		self.probabilities = {}
		for assignment in self.assignments:
			self.probabilities[assignment] = None
	def __str__(self):
		column_widths = [variable.column_width() for variable in self.variables]
		context = ''
		if len(self.context_assignment) > 0:
			context = '|{:}'.format(str(self.context_assignment)[1:-1])
		out_string = '{:} | P({:}{:})\n'.format(' | '.join([str(variable).ljust(column_widths[i]) for i, variable in enumerate(self.variables)]), ','.join([str(variable) for variable in self.variables]), context)
		out_string += '-'*len(out_string) + '\n'
		for assignment in self.assignments:
			for i, variable in enumerate(self.variables):
				out_string += str(assignment.get_variable(variable).value).ljust(column_widths[i]) + ' | '
			out_string += '{:}\n'.format(self.probabilities[assignment])
		return out_string[:-1]
	def __repr__(self):
		return str(self)
	def total_probability(self):
		return sum(self.probabilities.values())
	#def validate(self, epsilon=float_info.epsilon):
	def validate(self, epsilon=0.0000000000000004):
		if None in self.probabilities.values():
			return False
		if abs(1.0 - self.total_probability()) > epsilon:
			return False
		return True
	is_valid = property(validate)
	def __getitem__(self, key):
		assignment = Assignment(key)
		return self.get_row(assignment)
	def __setitem__(self, key, value):
		assignment = Assignment(key)
		return self.set_row(assignment, value)
	def get_row(self, assignment):
		return self.probabilities[assignment]
	def set_row(self, assignment, value):
		self.probabilities[assignment] = value
		return self
	def copy(self, other):
		if not self.variables == other.variables:
			raise KeyError('Cannot copy from marginal table that does not have the same variables.')
		for assignment in self.assignments:
			self.probabilities[assignment] = other.probabilities[assignment]
		return self
	def randomize(self):
		for assignment in self.assignments:
			self.probabilities[assignment] = random()
		self.normalize()
		return self
	def normalize(self):
		normalizer = sum(self.probabilities.values())
		for assignment in self.assignments:
			self.probabilities[assignment] /= normalizer
		return self
	def learn_from_complete_data(self, header, data):
		total_count = float(len(data))
		for assignment in self.assignments:
			self.probabilities[assignment] = 0
		for sample in data:
			sample_assignment = Assignment([SingleAssignment(variable, value) for variable, value in zip(header, sample)])
			self.probabilities[sample_assignment] += 1
		for assignment in self.assignments:
			self.probabilities[assignment] /= float(total_count)
		return self
	def learn_with_expectation_maximization(self, header, data, initial, max_iterations=1000):
		pass
	def marginalize_over(self, variables):
		return self.marginalize_out(self.variables.difference(set(variables)))
	def marginalize_out(self, variables):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like marginalization until marginal table is valid.')
		marginal = MarginalTable(self.variables.difference(set(variables)), self.context_assignment)
		marginalized_assignments = Assignment.generate(variables)
		for marginal_assignment in marginal.assignments:
			marginal.probabilities[marginal_assignment] = 0.0
		for marginal_assignment in marginal.assignments:
			for marginalized_assignment in marginalized_assignments:
				marginal.probabilities[marginal_assignment] += self.probabilities[marginal_assignment.union(marginalized_assignment)]
		return marginal.normalize()
	def condition(self, variables, context_variables):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like conditioning until marginal table is valid.')
		marginal = self.marginalize_over(set(variables).union(set(context_variables)))
		return marginal.condition_on(context_variables)
	def condition_on(self, context_variables):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like conditioning until marginal table is valid.')
		variables = self.variables.difference(set(context_variables))
		assignments = Assignment.generate(variables)
		conditional = ConditionalTable(variables, context_variables)
		context_marginal = self.marginalize_over(context_variables)
		for context_assignment in conditional.context_assignments:
			normalizer = context_marginal.probabilities[context_assignment]
			if normalizer == 0.0:
				raise ZeroDivisionError('Cannot condition due to deterministic (zero mass) probability: P{:} = 0.0'.format(context_assignment))
			context_table = conditional.context_tables[context_assignment]
			for assignment in assignments:
				context_table.probabilities[assignment] = self.probabilities[assignment.union(context_assignment)] / normalizer
			context_table.normalize()
		return conditional
	def get_samples(self, num_of_samples=1, header=None, as_assignment=False):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like sampling until marginal table is valid.')
		if header == None:
			header = list(self.variables)
		if as_assignment:
			choices = [assignment.ordered(header) for assignment in self.assignments]
		else:
			choices = [assignment.ordered_values(header) for assignment in self.assignments]
		weights = [self.probabilities[assignment] for assignment in self.assignments]
		weighted_choices = zip(weights, choices)
		return header, [weighted_choose(weighted_choices) for i in xrange(num_of_samples)]
	def __mul__(self, other):
		r'''
		Note that multiplying two marginals to get a marginal marginal is only
		valid if both marginals are independent of each other. We can see this
		from the definition of Bayes conditioning:

		$$P(a\mid b)=\frac{P(a,b)}{P(b)} \\
		P(a,b)=P(a\mid b)P(b)$$

		Note that this only holds when $P(a\mid b)=P(a)$ which is the case, by
		definition, when $A$ is independent of $B$.

		$$A\upmodels B\text{ iff }P(a\mid b)=P(a)$$
		'''
		if isinstance(other, MarginalTable):
			if len(self.variables.intersection(other.variables)) > 0:
				raise ValueError('Cannot multiply marginal tables because they share some variables: {:}'.format(', '.join([str(variable) for variable in self.variables.intersection(other.variables)])))
			variables = self.variables.union(other.variables)
			marginal = MarginalTable(variables)
			for assignment in marginal.assignments:
				marginal[assignment] = self[assignment.project(self.variables)] * other[assignment.project(other.variables)]
			return marginal
		raise NotImplementedError
	def __call__(self, *args):
		return self.query(*args)
	def query(self, *args):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like querying until marginal table is valid.')
		query, query_vars, given, given_vars, is_marginal_query, is_conditional_query, is_full_conditional_query = parse_query(*args)
		if is_conditional_query:
			if is_full_conditional_query:
				marginal = self.marginalize_over(query_vars + given_vars)
				return marginal.condition_on(given_vars)
			else:
				context_assignment = Assignment(given)
				conditional = self.condition_on(given_vars)
				marginal = conditional.context_tables[context_assignment]
		else:
			marginal = self

		marginal = marginal.marginalize_over(query_vars)
		if is_marginal_query:
			return marginal
		else:
			return marginal.probabilities[Assignment(query)]

class ConditionalTable():
	def __init__(self, variables, context_variables):
		self.variables = frozenset(variables)
		self.context_variables = frozenset(context_variables)
		self.all_variables = self.variables.union(self.context_variables)
		if len(self.variables.intersection(self.context_variables)) > 0:
			raise ValueError('Context variables and table variables cannot overlap: {:} exists in both {:} and {:}.'.format(self.variables.intersection(self.context_variables), self.variables, self.context_variables))
		self.assignments = Assignment.generate(self.variables)
		self.context_assignments = Assignment.generate(self.context_variables)
		self.all_assignments = Assignment.generate(self.variables.union(self.context_variables))
		self.context_tables = {}
		for context_assignment in self.context_assignments:
			self.context_tables[context_assignment] = MarginalTable(self.variables, context_assignment)
	def __str__(self):
		if len(self.context_variables) == 0:
			return str(self.context_tables[self.context_assignments[0]])
		context_column_widths = [variable.column_width() for variable in self.context_variables]
		column_widths = [variable.column_width() for variable in self.variables]
		out_string = '{:} || {:} | P({:}|{:})\n'.format(' | '.join([str(variable).ljust(context_column_widths[i]) for i, variable in enumerate(self.context_variables)]), ' | '.join([str(variable).ljust(column_widths[i]) for i, variable in enumerate(self.variables)]), ','.join([str(variable) for variable in self.variables]), ','.join([str(variable) for variable in self.context_variables]))
		out_string += '-'*len(out_string) + '\n'
		for context_assignment in self.context_assignments:
			context_table = self.context_tables[context_assignment]
			for assignment in context_table.assignments:
				out_string += ' | '.join([str(context_assignment.get_variable(variable).value).ljust(context_column_widths[i]) for i, variable in enumerate(self.context_variables)])
				out_string += ' || '
				out_string += ' | '.join([str(assignment.get_variable(variable).value).ljust(column_widths[i]) for i, variable in enumerate(self.variables)])
				out_string += ' | '
				out_string += '{:}\n'.format(context_table.probabilities[assignment])
		return out_string[:-1]
	def __repr__(self):
		return str(self)
	def validate(self):
		for assignment in self.context_assignments:
			if not self.context_tables[assignment].is_valid:
				return False
		return True
	is_valid = property(validate)
	def __getitem__(self, key):
		key = Assignment(key)
		assignment = key.project(self.variables)
		context_assignment = key.project(self.context_variables)
		return self.get_row(assignment, context_assignment)
	def __setitem__(self, key, value):
		key = Assignment(key)
		assignment = key.project(self.variables)
		context_assignment = key.project(self.context_variables)
		return self.set_row(assignment, context_assignment, value)
	def get_row(self, assignment, context_assignment):
		return self.context_tables[Assignment(context_assignment)].probabilities[Assignment(assignment)]
	def set_row(self, assignment, context_assignment, value):
		self.context_tables[Assignment(context_assignment)].set_row(Assignment(assignment), value)
		return self
	def randomize(self):
		for context_assignment in self.context_assignments:
			self.context_tables[context_assignment].randomize()
		return self
	def __mul__(self, other):
		'''
		'''
		if isinstance(other, MarginalTable):
			if not other.variables.issubset(self.context_variables):
				raise ValueError('Can only multiply a marginal table and conditional table if the marginal table variables are a subset of the conditional context variables: {:} is not a subset of {:}.'.format(', '.join([str(variable) for variable in other.variables]), ', '.join([str(variable) for variable in self.variables])))
			conditional = ConditionalTable(self.variables.union(other.variables), self.context_variables.difference(other.variables))
			for assignment in conditional.all_assignments:
				conditional[assignment] = self[assignment.project(self.all_variables)] * other[assignment.project(other.variables)]
			if len(conditional.context_variables) == 0:
				return conditional.context_tables[Assignment.empty]
			else:
				return conditional
		elif isinstance(other, ConditionalTable):
			# P(c|a,b)P(a|b)=P(c|b) by chain rule
			raise NotImplementedError
		raise NotImplementedError

################################################################################
# Graph constructs
################################################################################

class UndirectedEdge(tuple):
	def __str__(self):
		return '{:} - {:}'.format(str(self.from_var), str(self.to_var))
	def __repr__(self):
		return str(self)

BaseDirectedEdge = namedtuple('BaseDirectedEdge', ['from_var', 'to_var'])
class DirectedEdge(BaseDirectedEdge):
	def __str__(self):
		return '{:} > {:}'.format(str(self.from_var), str(self.to_var))
	def __repr__(self):
		return str(self)

def dag_parents(dag, variable):
	variables, edges = dag
	return frozenset(edge.from_var for edge in filter(lambda x: x.to_var == variable, edges))

def dag_children(dag, variable):
	variables, edges = dag
	return frozenset(edge.to_var for edge in filter(lambda x: x.from_var == variable, edges))

def dag_root_variables(dag):
	variables, edges = dag
	return frozenset(filter(lambda var: len(dag_parents(dag, var)) == 0, variables))

def dag_leaf_variables(dag):
	variables, edges = dag
	return frozenset(filter(lambda var: len(dag_children(dag, var)) == 0, variables))

def dag_topological_sort(dag):
	'''
	http://en.wikipedia.org/wiki/Topological_sorting
	'''
	variables, edges = dag
	order = []
	pending = list(dag_root_variables(dag))
	edges = set(edges)
	while len(pending) > 0:
		variable = pending.pop()
		order.append(variable)
		out_edges = filter(lambda x: x.from_var == variable, edges)
		edges.difference_update(out_edges)
		for edge in out_edges:
			if len(filter(lambda x: x.to_var == edge.to_var, edges)) == 0:
				pending.append(edge.to_var)
	if len(edges) > 0:
		raise ValueError('Graph has at least one cycle.')
	return order

BaseDirectedAcyclicGraph = namedtuple('BaseDirectedAcyclicGraph', ['variables', 'edges'])
class DirectedAcyclicGraph(BaseDirectedAcyclicGraph):
	def __init__(self, variables, edges):
		super(DirectedAcyclicGraph, self).__init__(frozenset(variables), frozenset(edges))
		self.root_variables = dag_root_variables(self)
		self.leaf_variables = dag_leaf_variables(self)
		self.families = dict(zip(self.variables, [dag_parents(self, variable) for variable in self.variables]))
		self.topological_order = dag_topological_sort(self)
	def __str__(self):
		return str(list(self.edges))
	def __repr__(self):
		return str(self)

################################################################################
# Graphical probabilistic models
################################################################################

class BayesianNetwork(DirectedAcyclicGraph):
	def __init__(self, variables, edges):
		super(BayesianNetwork, self).__init__(variables, edges)
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
	def simulate(self):
		sample = Assignment(())
		for variable in self.topological_order:
			conditional = self.conditionals[variable]
			context_assignment = sample.project(conditional.context_variables)
			context_table = conditional.context_tables[context_assignment]
			header, variable_samples = context_table.get_samples(as_assignment=True)
			sample = sample.union(variable_samples[0])
		return sample
	def get_samples(self, num_of_samples=1, header=None, as_assignment=False):
		if header == None:
			header = list(self.variables)
		if as_assignment:
			return header, [self.simulate().ordered(header) for i in xrange(num_of_samples)]
		else:
			return header, [self.simulate().ordered_values(header) for i in xrange(num_of_samples)]
	def as_marginal_table(self):
		marginal_table = MarginalTable(self.variables)
		for assignment in marginal_table.assignments:
			product = 1.0
			for variable in self.variables:
				conditional = self.conditionals[variable]
				product *= conditional.context_tables[assignment.project(self.families[variable])].probabilities[assignment.project([variable])]
			marginal_table.probabilities[assignment] = product
		return marginal_table.normalize()
	def get_display_js(self, width=640, height=480):
		with open('graph_display.js', 'r') as f:
			return 'var links=[{:}];var w={:},h={:};{:}'.format(''.join(['{{source:"{:}",target:"{:}"}},'.format(edge.from_var, edge.to_var) for edge in self.edges]), width, height, f.read())
	def display(self, width=640, height=480):
		import IPython.display
		# really hacky
		div_id = 'probgraphdisplay'+str(randint(0, 65536))
		IPython.display.display(IPython.display.HTML(data='<div id="{:}"></div>'.format(div_id)))
		IPython.display.display(IPython.display.Javascript(data='var cellDivId="{:}";{:}'.format(div_id, self.get_display_js(width, height)), lib='http://d3js.org/d3.v3.min.js', css='/files/graph_display.css'))
		return self

################################################################################
# Information theory
################################################################################

def entropy(distr, variables):
	ent = 0.0
	assignments = Assignment.generate(variables)
	for assignment in assignments:
		proba = distr(*assignment)
		if proba > 0.0:
			ent -= proba * log(proba, 2)
	return ent

def conditional_entropy(distr, X, Y):
	ent = 0.0
	for x in X.assignments:
		for y in Y.assignments:
			proba_xy = distr(x, y)
			proba_x_y = distr(x | y)
			if proba_xy > 0.0 and proba_x_y > 0.0:
				ent -= proba_xy * log(proba_x_y, 2)
	return ent

def mutual_information(distr, X, Y):
	mi = 0.0
	for x in X.assignments:
		for y in Y.assignments:
			proba_x = distr(x)
			proba_y = distr(y)
			proba_xy = distr(x, y)
			if proba_x > 0.0 and proba_y > 0.0 and proba_xy > 0.0:
				mi += proba_xy * log(proba_xy / (proba_x * proba_y), 2)
	return mi

def kl_divergence(distr_l, distr_r, variables):
	kl = 0.0
	assignments = Assignment.generate(variables)
	for assignment in assignments:
		proba_l = distr_l(*assignment)
		proba_r = distr_r(*assignment)
		if proba_l > 0.0 and proba_r > 0.0:
			kl += proba_l * log(proba_l / proba_r, 2)
	return kl
