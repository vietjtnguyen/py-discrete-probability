import collections
import math
import random
import sys

def entropy(distr, variables):
	ent = 0.0
	assignments = Assignment.generate(variables)
	for assignment in assignments:
		proba = distr(*assignment)
		if proba > 0.0:
			ent -= proba * math.log(proba, 2)
	return ent

def conditional_entropy(distr, X, Y):
	ent = 0.0
	for x in X.assignments:
		for y in Y.assignments:
			proba_xy = distr(x, y)
			proba_x_y = distr(x | y)
			if proba_xy > 0.0 and proba_x_y > 0.0:
				ent -= proba_xy * math.log(proba_x_y, 2)
	return ent

def mutual_information(distr, X, Y):
	mi = 0.0
	for x in X.assignments:
		for y in Y.assignments:
			proba_x = distr(x)
			proba_y = distr(y)
			proba_xy = distr(x, y)
			if proba_x > 0.0 and proba_y > 0.0 and proba_xy > 0.0:
				mi += proba_xy * math.log(proba_xy / (proba_x * proba_y), 2)
	return mi

def kl_divergence(distr_l, distr_r, variables):
	kl = 0.0
	assignments = Assignment.generate(variables)
	for assignment in assignments:
		proba_l = distr_l(*assignment)
		proba_r = distr_r(*assignment)
		if proba_l > 0.0 and proba_r > 0.0:
			kl += proba_l * math.log(proba_l / proba_r, 2)
	return kl

class Variable():
	def __init__(self, name, values=(True, False), description=''):
		self.name = name
		self.description = name if description == '' else description
		self.values = values
		self.assignments = Assignment([self<<value for value in self.values])
		if None in self.values:
			raise ValueError('Cannot use None as a value. None is reserved for missing data.')
	def get_assignments(self):
		return [self<<value for value in self.values]
	def __str__(self):
		return self.name
	def __repr__(self):
		return str(self)
	def column_width(self):
		return max(len(str(self)), max(*[len(str(value)) for value in self.values]))
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

BaseAssignment = collections.namedtuple('BaseAssignment', ['variable', 'value'])
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

class JointTable():
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
	def validate(self, epsilon=sys.float_info.epsilon):
		if None in self.probabilities.values():
			return False
		if abs(1.0 - sum(self.probabilities.values())) > epsilon:
			return False
		return True
	is_valid = property(validate)
	def set_row(self, assignment, value):
		self.probabilities[assignment] = value
		return self
	def copy(self, other):
		if not self.variables == other.variables:
			raise KeyError('Cannot copy from joint table that does not have the same variables.')
		for assignment in self.assignments:
			self.probabilities[assignment] = other.probabilities[assignment]
		return self
	def randomize(self):
		for assignment in self.assignments:
			self.probabilities[assignment] = random.random()
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
	def marginalize_over(self, variables):
		return self.marginalize_out(self.variables.difference(set(variables)))
	def marginalize_out(self, variables):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like marginalization until joint table is valid.')
		marginal = JointTable(self.variables.difference(set(variables)), self.context_assignment)
		marginalized_assignments = Assignment.generate(variables)
		for marginal_assignment in marginal.assignments:
			marginal.probabilities[marginal_assignment] = 0.0
		for marginal_assignment in marginal.assignments:
			for marginalized_assignment in marginalized_assignments:
				marginal.probabilities[marginal_assignment] += self.probabilities[marginal_assignment.union(marginalized_assignment)]
		return marginal
	def condition(self, variables, context_variables):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like conditioning until joint table is valid.')
		marginal = self.marginalize_over(set(variables).union(set(context_variables)))
		return marginal.condition_on(context_variables)
	def condition_on(self, context_variables):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like conditioning until joint table is valid.')
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
		return conditional
	def __call__(self, *args):
		if not self.is_valid:
			raise AssertionError('Cannot perform operations like querying until joint table is valid.')
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

		if is_conditional_query:
			is_full_conditional_query = len(filter(lambda x: isinstance(x, Variable), given)) > 0
			if is_full_conditional_query:
				marginal = self.marginalize_over(query_vars + given_vars)
				return marginal.condition_on(given_vars)
			else:
				context_assignment = Assignment(given)
				conditional = self.condition_on(given_vars)
				joint = conditional.context_tables[context_assignment]
		else:
			joint = self

		marginal = joint.marginalize_over(query_vars)
		if is_marginal_query:
			return marginal
		else:
			return marginal.probabilities[Assignment(query)]

class ConditionalTable():
	def __init__(self, variables, context_variables):
		self.variables = frozenset(variables)
		self.context_variables = frozenset(context_variables)
		if len(self.variables.intersection(self.context_variables)) > 0:
			raise ValueError('Context variables and table variables cannot overlap: {:} exists in both {:} and {:}.'.format(self.variables.intersection(self.context_variables), self.variables, self.context_variables))
		self.assignments = Assignment.generate(self.variables)
		self.context_assignments = Assignment.generate(self.context_variables)
		self.all_assignments = Assignment.generate(self.variables.union(self.context_variables))
		self.context_tables = {}
		for context_assignment in self.context_assignments:
			self.context_tables[context_assignment] = JointTable(self.variables, context_assignment)
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
	def set_row(self, assignment, context, value):
		self.context_tables[context].set_row(assignment, value)
		return self

class DirectedEdge():
	def __init__(self, from_var, to_var, right=True):
		self.from_var = from_var
		self.to_var = to_var
		self.right = right
	def __str__(self):
		return '{:} > {:}'.format(str(self.from_var), str(self.to_var))
	def __repr__(self):
		return str(self)

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
			if not self.parameterization[variable].is_valid:
				return False
		return True
	is_valid = property(validate)
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
			return 'var links=[{:}];{:}'.format(''.join(['{{source:"{:}",target:"{:}"}},'.format(edge.from_var, edge.to_var) for edge in self.edges]), f.read())
	def display(self, width=640, height=480):
		import IPython.display
		div_id = 'probgraphdisplay'+str(random.randint(0, 65536))
		IPython.display.display(IPython.display.HTML(data='<div id="{:}"></div>'.format(div_id)))
		IPython.display.display(IPython.display.Javascript(data='var cellDivId="{:}";{:}'.format(div_id, self.get_display_js(width, height)), lib='http://d3js.org/d3.v3.min.js', css='graph_display.css'))

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
