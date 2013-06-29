import collections
import math
import random
import sys

from variables import *

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
	def __getitem__(self, key):
		return self.get_row(key)
	def __setitem__(self, key, value):
		return self.set_row(key, value)
	def get_row(self, assignment):
		return self.probabilities[Assignment(assignment)]
	def set_row(self, assignment, value):
		self.probabilities[Assignment(assignment)] = value
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
	def direct_sample(self):
		raise NotImplementedError
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
	def __getitem__(self, key):
		assignment, context_assignment = key
		return self.get_row(assignment, context_assignment, value)
	def __setitem__(self, key, value):
		assignment, context_assignment = key
		return self.set_row(assignment, context_assignment, value)
	def get_row(self, assignment, context_assignment, value):
		return self.context_tables[Assignment(context_assignment)].probabilities[Assignment(assignment)]
	def set_row(self, assignment, context_assignment, value):
		self.context_tables[Assignment(context_assignment)].set_row(Assignment(assignment), value)
		return self
	def randomize(self):
		for context_assignment in self.context_assignments:
			self.context_tables[context_assignment].randomize()
		return self
	def direct_sample(self):
		raise NotImplementedError

