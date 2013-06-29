import collections.namedtuple
import math
import random
import sys

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

