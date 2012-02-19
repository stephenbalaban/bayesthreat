from itertools import product, combinations
from math import log

DEBUG = False

"""
Example BN with CPTs:
                 +-------+                      +-------+
    ( Burglary ) | P(B)  |      ( Earthquake )  | P(E)  |
        \        +-------+        /             +-------+
         \       | 0.001 |       /              | 0.002 |
          \      +-------+      /               +-------+
           \                   /
            \                 /                 +-----+-------+ 
             \-> ( Alarm ) <-/                  | B E | P(A)  |
                                                +-----+-------+
                 /      \                       | 1 1 | 0.95  |
                /        \                      +-----+-------+
               /          \                     | 1 0 | 0.94  |
              /            \                    +-----+-------+
             /              \                   | 0 1 | 0.29  |
            v                v                  +-----+-------+
    ( JohnCalls )              ( MaryCalls )    | 0 0 | 0.001 |
    +---+------+               +---+------+     +-----+-------+
    | A | P(J) |               | A | P(M) |
    +---+------+               +---+------+
    | 1 | 0.90 |               | 1 | 0.70 |
    +---+------+               +---+------+
    | 0 | 0.05 |               | 1 | 0.01 |
    +---+------+               +---+------+

    P(x_1, ..., x_n) = \Pi_{i=1]^n P(x_i | x_{i-1}, ..., x_n)
    Chain rule:
                     = \Pi_{i=1]^n P(x_i | Parents(X_i))

    P(!e, b, a, j, m) = P(j | a)P(m | a)P(a | !e, b)P(!e)P(b)
                      = 0.9 * 0.7 * 0.94 * (1 - 0.002) * 0.001
    P( b | j, m)

    P( X | e) = alpha P(X, e) = alpha \sum_y P(X, e, y)

BN with continuous variables:

    Not all problems can be described using boolean nodes. For example, if the
    amount of money spent on a financial transaction increases the probability
    of a fraud occurring, we might want to model the transaction amount as
    a continuous variable.

"""

class RV(object):
    def __init__(self, name,abbr):
        self.name = name
        self.abbr = abbr

def neg(pr):
    return 1 - pr

identity = lambda x: x

def sign_value(value, sign=True):
    return (identity if sign else neg)(value)

class BayesNode(object):
    def __init__(self, rv, cpt, parents=[], sign=True):
        self.sign = sign
        self.name = rv.name
        self.abbr = rv.abbr
        self.rv = rv
        self.parents = parents
        self.set_sign(sign)
        self.cpt = self.make_cpt(cpt, sign=sign)

    def set_sign(self, sign=True):
        self.name = self.name.replace('!', '') if sign else '!%s' % self.name if '!' not in self.name else self.name
        self.abbr = self.abbr.replace('!', '')  if sign else '!%s' % self.abbr if '!' not in self.abbr else self.abbr
        self.sign = sign
        return self

    def __repr__(self):
        result = "{ %s }" % self.name#\n%s\n" % (self.name, self.print_cpt(self.cpt))
        result = "{ %s }\n%s\n" % (self.name, self.print_cpt(self.cpt))
        return result

    def __nonzero__(self):
        return bool(self.sign)

    def get_cpt(self, key):
        return sign_value(self.cpt[key], self.sign)

    def neg(self):
        return self.set_sign(not(self.sign))

    def pr(self, rvs=(), given=(), full_dist={}):
        """
        No priors:
            P(x_1, ..., x_n) = \Pi_{i=1]^n P(x_i | x_{i-1}, ..., x_n)
        With Priors (loop over hidden variables):
            Full distribution: 
                P(x_1, x_2, x_3, x_4, x_5, ..., x_n)
            Prior probabilities:
                P( x_1, x_2 | x_3) = \alpha P(x_1, x_2, x_3) 
                                   = \alpha \Sigma_{x_4} \Sigma_{x_5} ... \Sigma_{x_n} P(x_1, x_2, ..., x_n)
        """
        if given:
            pr = self.calculate_prior_pr(rvs, given, full_dist=full_dist)
        else:
            pr = self.calculate_full_pr(rvs)
        return pr

    def rv_possibilities(cls, full_distribution, hidden_variables):
        """ 
        maps full_dist, hidden_vars -> all possible full distributions
        iterating over the values of hidden variables (requires boolean rvs)
        """
        set_difference = list(set(full_distribution.values()) - set(hidden_variables))
        print("set diff: %s" % set_difference)
        # all possible characters to combine
        possibilities = flatten([[hv, hv.neg()] for hv in hidden_variables])
        # all combinations of those characters
        combos = make_combinations(possibilities, len(hidden_variables)/2 -1, splitchar='!')
        # those combos consed into a new list
        final_combos = [combo + set_difference for combo in combos]
        print("""

        hidden vars:
        %s

        set diff:
        %s
        
    -------------------
        final combos:
        %s
    -------------------
        combols: 
        %s
    -------------------
        possibilitit
        %s
        
    -------------------
        
        """ % (hidden_variables, set_difference, final_combos, combos, possibilities))
        return final_combos

    def calculate_prior_pr(self, rvs, given, full_dist):
        """
        """
        hidden_vars = list(set(full_dist.values()) - set(rvs + given))
        print('fd %s ' % full_dist)
        print("hv %s " % (hidden_vars))
        return sum([
                self.calculate_full_pr(rvs) for rvs in self.rv_possibilities(full_dist, hidden_vars)
        ])
        pass

    def calculate_full_pr(self, rvs):
        """
        """
        if self in rvs:
            this_node = filter(lambda n: n == self, rvs)[0]
        else:
            print(self)
            print('rvs: %s' % rvs)
            raise Exception("Couldn't use the rvs given... didn't find myself there!")
        this_parents = list(filter(lambda n: n in self.parents, rvs))
        if not this_parents:
            this_parents_key = True
        else:
            this_parents_key = tuple([bool(p) for p in this_parents]) if len(this_parents) > 1 else bool(this_parents[0])
        pr = self.get_cpt(this_parents_key)
        if DEBUG:
            print("""
            this_node: %s 
            this_node.sign: %s
            this_parents:    %s
            this_parents_key: %s
            this_cpt: %s
            """ % (this_node, this_node.sign, this_parents, this_parents_key, self.cpt))
            print(pr)
        return pr

    def print_cpt(self, cpt):
        """
        A large pile of crufty hacks to print out the CPTs, string formatting
        is fun!
        """
        rows = [self.print_row(item) for item in cpt.items()]
        header_len = len(rows[0].split('\n')[0]) - 2
        between = '+' + '-' * header_len + '+\n'
        if self.parents:
            header = '| ' + ''.join(['%s ' for p in self.parents]) + '| P(%s)  |\n' % self.abbr
            header = header % tuple([p.abbr for p in self.parents])
            result = '\n'.join([self.print_row(item) for item in cpt.items()])
        else:
            header = ('| P(%s)' + ' ' * (header_len - 5) + '|\n') % self.abbr
            result = '\n'.join([self.print_row(item, no_parents=True) for item in cpt.items()])
        return between + header + between + result

    def print_row(self, cpt_item, no_parents=False):
        """
        maps tuple like: 
            (True, 0.95) -> | 0.95 |
            ((True, False, True), 0.90) -> | 1 0 1 | 0.90 | 
        """
        cpt_item = list(cpt_item)
        # make body
        if type(cpt_item[0]) in [tuple, list]:
            cpt_item[0] = map(int, cpt_item[0])
            row = '| ' + ''.join(['%s ' for var in cpt_item[0]]) + '|'
            row = row % tuple(cpt_item[0])
        else:
            cpt_item[0] = int(cpt_item[0])
            row = '| %s |' % cpt_item[0]
        if no_parents:
            row = '| %s     |' % cpt_item[1]
        else:
            row += ' %.3f |' % cpt_item[1]
        # add rows in between
        row += '\n+' + '-' * (len(row) - 2) + '+'
        return row

    def make_cpt(self, cpt, sign=True):
        cpt_type = type(cpt)
        if cpt_type not in [list, dict]:
            raise Exception("Invalid CPT given, type(cpt): %s" % cpt_type)
        f = self.make_dict_cpt if cpt_type == dict else self.make_list_cpt
        r = f(cpt)
        # negate values in dictionary if sign
        return dict([(key, sign_value(val, sign)) for key, val in r.items()])

    def make_dict_cpt(self, cpt):
        result = {}
        for k in cpt:
            result[bool(k)] = cpt[k]
        return result

    def var_count(self, cpt_list):
        return int(log(len(cpt_list), 2))

    def make_list_cpt(self, cpt_list):
        """ maps cpt_list [ pr, pr, pr ] -> dictionary  """
        n = self.var_count(cpt_list)
        keys = product([True,False], repeat=n) if n > 1 else [True, False]
        result = dict(zip(keys, cpt_list))
        return result


def unique(splitchar="!"):
    def result(tup):
        return len(set(map(lambda x: x.split(splitchar)[-1], tup))) == len(map(lambda x: x.split(splitchar)[-1], tup))
    return result

def flatten(l):
    return [item for sublist in l for item in sublist]

def make_combinations(possibilities, count, splitchar='!'):
    return map(list, filter(unique(splitchar), list(combinations(possibilities, count))))

class BayesNet(object):
    def __init__(self, *nodes):
        self.nodes = dict(map(lambda n: (n.name, n), nodes))
        self.abbrs = dict(map(lambda n: (n.abbr, n), nodes))

    def __repr__(self):
        result = "BayesNet\n"
        result += '\n'.join(map(str, self.nodes.values()))
        return result

    def get(self, abbr):
        return self.abbrs[abbr]

    def name(self, name):
        return self.nodes[name]

    def pr(self, string):
        """
           returns  Pr( rv | given )
        """
        rvs, given = self.parse_query(string)
        return capital_pi([node.pr(rvs, given, full_dist=self.nodes) for node in self.nodes.values()])

    def parse_query(self, query):
        """
        maps string query -> rv, given list
        """
        if '|' in query:
            dirty_rvs, dirty_given = query.split('|')
            given_strs = tuple([x.strip() for x in dirty_given.split(',')])
        else:
            dirty_rvs = query.split(',')
            given_strs = tuple([])
        rvs = self.parse_rvs(dirty_rvs)
        given = self.parse_given(given_strs)
        return (rvs, given)

    def abbr_pos_nodes(self, abbr_positive):
        # generate nodes
        print(abbr_positive)
        rvs = [self.abbrs[abbr].set_sign(positive) for abbr, positive in abbr_positive]
        return tuple(rvs)

    def parse_given(self, given_strs):
        """ parses given strs into given rvs """
        abbr_positive = [self.parse_string(rv) for rv in given_strs]
        return tuple(self.abbr_pos_nodes(abbr_positive))

    def parse_rvs(self, dirty_rvs):
        """ parses dirty rvs to a list of rvs """
        # remove spaces
        rvs = [x.strip() for x in dirty_rvs]
        # remove empty strings
        rvs = filter(bool, rvs)
        abbr_positive = [self.parse_string(rv) for rv in rvs]
        return tuple(self.abbr_pos_nodes(abbr_positive))

    def parse_string(self, rv_str):
        """
        maps node -> node_string, positive?
             !a -> a, False
             c  -> c, True
        """
        return (rv_str.replace('!', ''), '!' not in rv_str)

capital_pi = lambda l: reduce(lambda a,b: a*b, l)

def make_earthquake_model():
    erv = RV(name="Earthquake", abbr='e')
    brv = RV("Burgler", 'b')
    arv = RV("Alarm", 'a')
    mrv = RV("MaryCalls", 'm')

    jrv = RV("JohnCalls", 'j')
    e = BayesNode(rv=erv, cpt=[0.002], parents=[]) # [0.002] expands to { 1: 0.002, 0: 1 - 0.002 }
    b = BayesNode(brv, cpt={ True: 0.001 }) # no parentsj
    a = BayesNode(arv, cpt=[0.95, 0.94, 0.29, 0.001], parents=[b, e])
    m = BayesNode(mrv, cpt=[0.70, 0.01], parents=[a]) # [0.70, 0.01] expands to { 1: 0.70, 0: 0.01 }
    j = BayesNode(jrv, cpt=[0.90, 0.05], parents=[a])
    bn = BayesNet(e,b,a,j,m)
    return bn


def run():
    make_earthquake_model()

if __name__ == '__main__':
    run()

