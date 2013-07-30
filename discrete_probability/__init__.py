__version__ = '0.10dev-8f143b8'

from collections import namedtuple
from decimal import Decimal
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

def data_to_assignments(header, samples):
    sample_assignments = []
    for sample in samples:
        single_assignment_tuples = filter(lambda x: not x[1] == None, zip(header, sample))
        single_assignments = [SingleAssignment(variable, value) for variable, value in single_assignment_tuples]
        sample_assignment = Assignment(single_assignments)
        sample_assignments.append(sample_assignment)
    return sample_assignments

class Query():

    def __init__(self, query, given=[]):
        self._query = query
        self._given = given
        self._update()

    def __str__(self):
        return '{:} | {:}'.format(', '.join([str(x) for x in self._query]), ', '.join([str(x) for x in self._given]))

    def __repr__(self):
        return str(self)

    def get_query(self):
        return self._query

    def set_query(self, value):
        self._query = value
        self._update()

    query = property(get_query, set_query)

    def get_given(self):
        return self._given

    def set_given(self, value):
        self._given = value
        self._update()

    given = property(get_given, set_given)

    def _update(self):
        self.query_vars = map(lambda x: x if isinstance(x, Variable) else x.variable, self._query)
        self.given_vars = map(lambda x: x if isinstance(x, Variable) else x.variable, self._given)
        self.is_marginal_query = len(filter(lambda x: isinstance(x, Variable), self._query)) > 0
        self.is_conditional_query = len(self._given) > 0
        self.is_full_conditional_query = len(filter(lambda x: isinstance(x, Variable), self.given)) > 0

    @staticmethod
    def from_natural(*args):
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
        return Query(query, given)

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

    def complete_partials(self, variables):
        return Assignment.generate(set(variables).difference(self.get_variables()), [])

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
# Number systems
################################################################################

float_number_system = (0.0, 1.0, float)
decimal_number_system = (Decimal('0'), Decimal('1'), Decimal)
try:
    from sympy import S
    sympy_number_system = (S('0'), S('1'), S)
    sympy_float_number_system = (S('0.0'), S('1.0'), S)
except ImportError:
    pass

################################################################################
# Discrete probability tables
################################################################################

# TODO: These tables should really be numeric arrays with a function to
# transform Assignments to indices.
# TODO: All of these if statements to perform a different operation depending
# on whether the table is marginal or not might be best done by simply
# reassigning the methods on initialization since a table cannot change its own
# type.
class Table():

    def __init__(self, variables, context_variables=(), context='', ignore_validity=False, number_system=float_number_system):
        self.variables = frozenset(variables)
        self.context_variables = frozenset(context_variables)
        self.all_variables = self.variables.union(self.context_variables)

        self.context = context
        if self.context == '':
            self.context = str(tuple(self.context_variables))[1:-1]
        self.ignore_validity = ignore_validity
        self.set_number_system(number_system)

        if len(self.variables.intersection(self.context_variables)) > 0:
            raise ValueError('Context variables and table variables cannot overlap: {:} exists in both {:} and {:}.'.format(self.variables.intersection(self.context_variables), self.variables, self.context_variables))

        self.is_conditional_table = len(self.context_variables) > 0
        self.is_marginal_table = len(self.context_variables) == 0

        self.assignments = Assignment.generate(self.variables)
        self.context_assignments = Assignment.generate(self.context_variables)
        self.all_assignments = Assignment.generate(self.variables.union(self.context_variables))

        self._entries = {}
        if self.is_marginal_table:
            for assignment in self.assignments:
                self._entries[assignment] = None
        else:
            for context_assignment in self.context_assignments:
                self._entries[context_assignment] = Table(self.variables, (), str(context_assignment)[1:-1], ignore_validity=self.ignore_validity, number_system=self.number_system)

    # REPRESENTATIONS

    def __str__(self):
        context_column_widths = [variable.column_width() for variable in self.context_variables]
        column_widths = [variable.column_width() for variable in self.variables]
        if self.is_marginal_table:
            out_string = '{:} | P({:}{:})\n'.format(
                ' | '.join([str(variable).ljust(column_widths[i]) for i, variable in enumerate(self.variables)]),
                ','.join([str(variable) for variable in self.variables]),
                '|{:}'.format(self.context) if not self.context == '' else '')
            out_string += '-'*len(out_string) + '\n'
            for assignment in self.assignments:
                for i, variable in enumerate(self.variables):
                    out_string += str(assignment.get_variable(variable).value).ljust(column_widths[i]) + ' | '
                out_string += '{:}\n'.format(self._entries[assignment])
        else:
            out_string = '{:} || {:} | P({:}|{:})\n'.format(' | '.join([str(variable).ljust(context_column_widths[i]) for i, variable in enumerate(self.context_variables)]), ' | '.join([str(variable).ljust(column_widths[i]) for i, variable in enumerate(self.variables)]), ','.join([str(variable) for variable in self.variables]), ','.join([str(variable) for variable in self.context_variables]))
            out_string += '-'*len(out_string) + '\n'
            for context_assignment in self.context_assignments:
                context_table = self._entries[context_assignment]
                for assignment in context_table.assignments:
                    out_string += ' | '.join([str(context_assignment.get_variable(variable).value).ljust(context_column_widths[i]) for i, variable in enumerate(self.context_variables)])
                    out_string += ' || '
                    out_string += ' | '.join([str(assignment.get_variable(variable).value).ljust(column_widths[i]) for i, variable in enumerate(self.variables)])
                    out_string += ' | '
                    out_string += '{:}\n'.format(context_table[assignment])
        return out_string[:-1]

    def __repr__(self):
        return str(self)

    # BASIC OPERATIONS

    def __getitem__(self, key):
        key = Assignment(key)
        assignment = key.project(self.variables)
        context_assignment = key.project(self.context_variables)
        if self.is_marginal_table:
            return self._entries[assignment]
        else:
            return self._entries[context_assignment][assignment]

    def __setitem__(self, key, value):
        key = Assignment(key)
        assignment = key.project(self.variables)
        context_assignment = key.project(self.context_variables)
        if self.is_marginal_table:
            self._entries[assignment] = value
        else:
            self._entries[context_assignment][assignment] = value

    def get_number_system(self):
        return self.zero, self.one, self.make_number

    def set_number_system(self, value):
        self.zero, self.one, self.make_number = value

    number_system = property(get_number_system, set_number_system)

    def total_probability(self):
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')
        return sum(self._entries.values())

    def get_is_valid(self, epsilon=0.0000000000000004):
        if self.is_marginal_table:
            if None in self._entries.values():
                return False
            if epsilon == 0 or self.make_number != float:
                if not self.total_probability() == self.one:
                    return False
            else:
                if abs(self.one - self.total_probability()) > epsilon:
                    return False
        else:
            for context_assignment in self.context_assignments:
                if not self._entries[context_assignment].is_valid:
                    return False
        return True

    is_valid = property(get_is_valid)

    def copy(self, other):
        '''
        Copies the entries from another table. This requires that the other
        table share the same variables as this one.

        Raises a KeyError if the tables do not share the same variables.
        '''
        if not self.variables == other.variables and self.context_variables == other.context_variables:
            raise KeyError('Cannot copy from table that does not have the same variables.')
        if self.is_marginal_table:
            for assignment in self.assignments:
                self._entries[assignment] = other._entries[assignment]
        else:
            for context_assignment in self.context_assignments:
                self._entries[context_assignment].copy(other._entries[context_assignment])
        return self

    def randomize(self):
        '''
        Randomizes the table entries.

        If the table is a marginal table then it assigns a random value from a
        uniform distribution in [0, 1] to each entry and then normalizes the
        table.

        If the table is a conditional table then it simply calls randomize on
        all context tables.

        The function returns the instance.
        '''
        if self.is_marginal_table:
            for assignment in self.assignments:
                self._entries[assignment] = random()
        else:
            for context_assignment in self.context_assignments:
                self._entries[context_assignment].randomize()
        self.normalize()
        return self
    
    def normalize(self):
        '''
        Normalizes the table.
        
        If the table is a marginal table then it simply calculates the total
        probablity and divides all entries by the total probability.

        If the table is a conditional table then it simply calls normalize on
        all context tables.

        The function returns the instance.
        '''
        if self.is_marginal_table:
            normalizer = self.total_probability()
            for assignment in self.assignments:
                self._entries[assignment] /= normalizer
        else:
            for context_assignment in self.context_assignments:
                self._entries[context_assignment].normalize()
        return self
    
    def normalize_number_system(self):
        if self.is_marginal_table:
            for assignment in self.assignments:
                self._entries[assignment] = self.make_number(self._entries[assignment])
        else:
            for context_assignment in self.context_assignments:
                self._entries[context_assignment].normalize()
        return self

    # BASIC PROBABILISTIC OPERATIONS
        
    def marginalize_out(self, variables):
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')
        if not self.is_valid and not self.ignore_validity:
            raise AssertionError('Cannot perform operations like marginalization until marginal table is valid.')
        marginal = Table(self.variables.difference(set(variables)), context=self.context, ignore_validity=self.ignore_validity, number_system=self.number_system)
        marginalized_assignments = Assignment.generate(variables)
        for marginal_assignment in marginal.assignments:
            marginal._entries[marginal_assignment] = self.zero
        for marginal_assignment in marginal.assignments:
            for marginalized_assignment in marginalized_assignments:
                marginal._entries[marginal_assignment] += self._entries[marginal_assignment.union(marginalized_assignment)]
        return marginal.normalize()

    def marginalize_over(self, variables):
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')
        return self.marginalize_out(self.variables.difference(set(variables)))

    def condition_on(self, context_variables):
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')
        if not self.is_valid and not self.ignore_validity:
            raise AssertionError('Cannot perform operations like conditioning until marginal table is valid.')
        variables = self.variables.difference(set(context_variables))
        assignments = Assignment.generate(variables)
        conditional = Table(variables, context_variables, ignore_validity=self.ignore_validity, number_system=self.number_system)
        context_marginal = self.marginalize_over(context_variables)
        for context_assignment in conditional.context_assignments:
            normalizer = context_marginal._entries[context_assignment]
            if normalizer == self.zero:
                raise ZeroDivisionError('Cannot condition due to deterministic (zero mass) probability: P{:} = 0'.format(context_assignment))
            context_table = conditional._entries[context_assignment]
            for assignment in assignments:
                context_table._entries[assignment] = self._entries[assignment.union(context_assignment)] / normalizer
            # TODO: Is this one too many normalizations?
            #context_table.normalize()
        return conditional

    def condition(self, variables, context_variables):
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')
        if not self.is_valid and not self.ignore_validity:
            raise AssertionError('Cannot perform operations like conditioning until marginal table is valid.')
        marginal = self.marginalize_over(set(variables).union(set(context_variables)))
        return marginal.condition_on(context_variables)

    def chain_rule(self, others):
        raise NotImplementedError

    def __mul__(self, other):
        r'''
        Note that multiplying two marginals to get a marginal is only
        valid if both marginals are independent of each other. We can see this
        from the definition of Bayes conditioning:

        $$P(a\mid b)=\frac{P(a,b)}{P(b)} \\
        P(a,b)=P(a\mid b)P(b)$$

        Note that this only holds when $P(a\mid b)=P(a)$ which is the case, by
        definition, when $A$ is independent of $B$.

        $$A\upmodels B\text{ iff }P(a\mid b)=P(a)$$
        '''
        if self.is_marginal_table and other.is_marginal_table:
            if len(self.variables.intersection(other.variables)) > 0:
                raise ValueError('Cannot multiply marginal tables because they share some variables: {:}'.format(', '.join([str(variable) for variable in self.variables.intersection(other.variables)])))
            variables = self.variables.union(other.variables)
            marginal = Table(variables, ignore_validity=self.ignore_validity, number_system=self.number_system)
            for assignment in marginal.assignments:
                marginal[assignment] = self[assignment.project(self.variables)] * other[assignment.project(other.variables)]
            return marginal
        raise NotImplementedError

    # QUERIES

    def query(self, query):
        if not self.is_valid and not self.ignore_validity:
            raise AssertionError('Cannot perform operations like querying until marginal table is valid.')
        if self.is_marginal_table:
            if query.is_conditional_query:
                if query.is_full_conditional_query:
                    marginal = self.marginalize_over(query.query_vars + query.given_vars)
                    return marginal.condition_on(query.given_vars)
                else:
                    context_assignment = Assignment(query.given)
                    conditional = self.condition_on(query.given_vars)
                    marginal = conditional._entries[context_assignment]
            else:
                marginal = self

            marginal = marginal.marginalize_over(query.query_vars)
            if query.is_marginal_query:
                return marginal
            else:
                return marginal._entries[Assignment(query.query)]
        # TODO: Could implement queries on conditional tables if the given
        # variables of the query are equal to the conditional table.
        raise NotImplementedError

    def __call__(self, *args):
        return self.query(Query.from_natural(*args))

    # DATA METRICS

    def data_likelihood(self, sample_assignments):
        '''
        Returns the likelihood of the data. The data is expected to be an
        enumerable of Assignments.
        '''
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')
        likelihood = self.one
        sample_likelihoods = []
        for sample_assignment in sample_assignments:
            query = Query(sample_assignment)
            sample_likelihood = self.query(query)
            sample_likelihoods.append(sample_likelihood)
            likelihood *= sample_likelihood
        return likelihood

    def data_log_likelihood(self, sample_assignments):
        '''
        Returns the log likelihood of the data. The data is expected to be an
        enumerable of Assignments.
        '''
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')
        log_likelihood = self.zero
        sample_log_likelihoods = []
        for sample_assignment in sample_assignments:
            query = Query(sample_assignment)
            sample_log_likelihood = log(self.query(query))
            sample_log_likelihoods.append(sample_log_likelihood)
            log_likelihood += sample_log_likelihood
        return log_likelihood

    # SAMPLING

    def get_samples(self, num_of_samples=1, header=None, as_assignment=False):
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')
        if not self.is_valid:
            raise AssertionError('Cannot perform operations like sampling until marginal table is valid.')
        if header == None:
            header = list(self.variables)
        if as_assignment:
            choices = [assignment.ordered(header) for assignment in self.assignments]
        else:
            choices = [assignment.ordered_values(header) for assignment in self.assignments]
        weights = [self._entries[assignment] for assignment in self.assignments]
        weighted_choices = zip(weights, choices)
        return header, [weighted_choose(weighted_choices) for i in xrange(num_of_samples)]

    # PARAMETER LEARNING

    def learn_from_complete_data(self, sample_assignments):
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')

        total_count = self.make_number(len(sample_assignments))

        for assignment in self.assignments:
            self._entries[assignment] = self.zero
        for sample_assignment in sample_assignments:
            self._entries[sample_assignment] += self.one
        for assignment in self.assignments:
            self._entries[assignment] /= total_count

        return self

    def learn_with_expectation_maximization(self, sample_assignments, initial, max_iterations=1000, log_likelihood_threshold = 0.000001): # TODO: Figure out what an appropriate threshold is, maybe difference in distribution instead? (Viet Nguyen, 2013-07-09)
        if not self.is_marginal_table:
            raise ValueError('Operation only valid for marginal tables.')

        total_count = self.make_number(len(sample_assignments))

        # Set the table to the initial parameter estimates
        for assignment in self.assignments:
            self._entries[assignment] = initial.query(Query(assignment))

        accumulators = {}
        last_log_likelihood = self.data_log_likelihood(sample_assignments)
        iter_count = 0

        while iter_count < max_iterations:
            for assignment in self.assignments:
                accumulators[assignment] = self.zero

            for sample_assignment in sample_assignments:
                if sample_assignment.get_variables() == self.variables:
                    # NOTE: This branch is a (possibly premature) optimization. The algorithm would work using just the else branch.
                    accumulators[sample_assignment] += self.one
                else:
                    completed_assignment_partials = sample_assignment.complete_partials(self.variables)
                    for assignment_partial in completed_assignment_partials:
                        query = Query(assignment_partial, sample_assignment)
                        query_result = self.query(query)
                        completed_assignment = sample_assignment.union(assignment_partial)
                        accumulators[completed_assignment] += query_result

            for assignment in self.assignments:
                self._entries[assignment] = accumulators[assignment] / total_count

            log_likelihood = self.data_log_likelihood(sample_assignments)
            iter_count += 1

            if log_likelihood - last_log_likelihood < log_likelihood_threshold:
                # NOTE: This difference should theoretically be always positive (see Corollary 11 in section 17.3 of Darwiche)
                break
            last_log_likelihood = log_likelihood

        return self

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

class BayesianNetwork():

    def __init__(self, variables, edges, ignore_validity=False, number_system=float_number_system):
        self.variables = variables
        self.edges = edges
        self.dag = DirectedAcyclicGraph(self.variables, self.edges)
        self.ignore_validity = ignore_validity
        self.set_number_system(number_system)
        self.conditionals = {}
        for variable in self.variables:
            self.conditionals[variable] = Table([variable], self.dag.families[variable], ignore_validity=self.ignore_validity, number_system=self.number_system)

    def get_number_system(self):
        return self.zero, self.one, self.make_number

    def set_number_system(self, value):
        self.zero, self.one, self.make_number = value

    number_system = property(get_number_system, set_number_system)

    def get_is_valid(self):
        for variable in self.variables:
            if not self.conditionals[variable].is_valid:
                return False
        return True
    is_valid = property(get_is_valid)

    def randomize(self):
        for conditional in self.conditionals.values():
            conditional.randomize()
        return self

    def learn_from_complete_data(self, sample_assignments):
        total_count = self.make_number(len(sample_assignments))
        accum_assignments = set()
        for variable in self.variables:
            conditional = self.conditionals[variable]
            accum_assignments.update(conditional.all_assignments)
            accum_assignments.update(conditional.context_assignments)
            for assignment in conditional.all_assignments:
                conditional[assignment] = self.zero
        accumulators = {}
        for accum_assignment in accum_assignments:
            accumulators[accum_assignment] = self.zero
        for accum_assignment in accum_assignments:
            for sample_assignment in sample_assignments:
                if accum_assignment.consistent_with(sample_assignment):
                    accumulators[accum_assignment] += self.one
        for variable in self.variables:
            conditional = self.conditionals[variable]
            for context_assignment in conditional.context_assignments:
                for assignment in conditional.assignments:
                    conditional[assignment.union(context_assignment)] = accumulators[assignment.union(context_assignment)]/accumulators[context_assignment]
        return self

    def simulate(self):
        sample = Assignment(())
        for variable in self.dag.topological_order:
            conditional = self.conditionals[variable]
            context_assignment = sample.project(conditional.context_variables)
            context_table = conditional._entries[context_assignment]
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

    def as_marginal_table(self, **kwargs):
        marginal_table = Table(self.variables, **kwargs)
        for assignment in marginal_table.assignments:
            product = self.one
            for variable in self.variables:
                conditional = self.conditionals[variable]
                product *= conditional[assignment]
            marginal_table[assignment] = product
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

# TODO: Expand number system to also specify log functions and such

def entropy(distr, variables, number_system=float_number_system):
    zero, one, make_number = number_system
    ent = zero
    assignments = Assignment.generate(variables)
    for assignment in assignments:
        proba = distr(*assignment)
        if proba > zero:
            ent -= proba * log(proba, one+one)
    return ent

def conditional_entropy(distr, X, Y, number_system=float_number_system):
    zero, one, make_number = number_system
    ent = zero
    for x in X.assignments:
        for y in Y.assignments:
            proba_xy = distr(x, y)
            proba_x_y = distr(x | y)
            if proba_xy > zero and proba_x_y > zero:
                ent -= proba_xy * log(proba_x_y, one+one)
    return ent

def mutual_information(distr, X, Y, number_system=float_number_system):
    zero, one, make_number = number_system
    mi = zero
    for x in X.assignments:
        for y in Y.assignments:
            proba_x = distr(x)
            proba_y = distr(y)
            proba_xy = distr(x, y)
            if proba_x > zero and proba_y > zero and proba_xy > zero:
                mi += proba_xy * log(proba_xy / (proba_x * proba_y), one+one)
    return mi

def kl_divergence(distr_l, distr_r, variables, number_system=float_number_system):
    zero, one, make_number = number_system
    kl = zero
    assignments = Assignment.generate(variables)
    for assignment in assignments:
        proba_l = distr_l(*assignment)
        proba_r = distr_r(*assignment)
        if proba_l > zero and proba_r > zero:
            kl += proba_l * log(proba_l / proba_r, one+one)
    return kl

