from itertools import product
from math import log

"""
Example BN with CPTs:
                 +-------+                      +-------+
    ( Burglary ) | P(B)  |      ( Earthquake )  | P(E)  |
        \        +-------+        /             +-------+
         \       | 0.001 |       /              | 0.002 |
          \      +-------+      /               +-------+
           \                   /
            \                 /                 +-------+-------+ 
             \-> ( Alarm ) <-/                  | B  E  | P(A)  |
                                                +-------+-------+
                 /      \                       | 1  1  | 0.95  |
                /        \                      +-------+-------+
               /          \                     | 1  0  | 0.94  |
              /            \                    +-------+-------+
             /              \                   | 0  1  | 0.29  |
            v                v                  +-------+-------+
    ( JohnCalls )              ( MaryCalls )    | 0  0  | 0.001 |
    +---+------+               +---+------+     +-------+-------+
    | A | P(J) |               | A | P(M) |
    +---+------+               +---+------+
    | 1 | 0.90 |               | 1 | 0.70 |
    +---+------+               +---+------+
    | 0 | 0.05 |               | 1 | 0.01 |
    +---+------+               +---+------+

    P(x_1, ..., x_n) = \Pi_{i=1]^n P(x_i | x_{i-1}, ..., x_n)
    Chain rule:
                     = \Pi_{i=1]^n P(x_i | Parents(X_i))

BN with continuous variables:

    Not all problems can be described using boolean nodes. For example, if the
    amount of money spent on a financial transaction increases the probability
    of a fraud occurring, we might want to model the transaction amount as
    a continuous variable.

"""

class Dag(object):
    """
    A Bayes Net is a Directed Acyclic Graph (DAG), represented as an adjacency
    matrix
    """
    def __init__(self, *nodes):
        self.matrix = make_matrix(nodes)
        pass


class RV(object):
    def __init__(self, name,abbr):
        self.name = name
        self.abbr = abbr

class BayesNode(object):
    def __init__(self, rv, cpt, parents=[]):
        self.name = rv.name
        self.abbr = rv.abbr
        self.rv = rv
        self.parents = parents
        self.cpt = self.make_cpt(cpt)
    def __repr__(self):
        result = "( %s )" % self.name
        for row in self.cpt:
            result += str(row)
        return result
    def make_cpt(self, cpt):
        cpt_type = type(cpt)
        if cpt_type not in [list, dict]:
            raise Exception("Invalid CPT given, type(cpt): %s" % cpt_type)
        f = self.make_dict_cpt if cpt_type == dict else self.make_list_cpt
        return f(cpt)
    def make_dict_cpt(self, cpt):
        result = {}
        for k in cpt:
            result[bool(k)] = cpt[k]
        if len(cpt) == 1: # binary RV
            key = cpt.keys()[0]
            value = cpt.values()[0]
            result[bool(key)] = value
            result[not key] = 1 - value
        return result
    def make_list_cpt(self, cpt_list):
        """ maps cpt_list [ pr, pr, pr ] -> dictionary  """
        n = int(log(len(cpt_list), 2)) # the number of varsz
        keys = product([True,False], repeat=n) if n > 2 else [1, 0]
        if len(cpt_list) == 1:
            cpt_list.append(1 - cpt_list[0])
        return dict(zip(keys, cpt_list))

class BayesNet(object):
    def __init__(self, *nodes):
        self.nodes = dict(map(lambda n: (n.name, n), nodes))
        self.abbrs = dict(map(lambda n: (n.abbr, n), nodes))
    def __repr__(self):
        return '\n'.join(str(self.nodes))
    def pr(*rvs):
        """
        Calculates the probability of the given RVs, e.g.
            P(x_1, ..., x_n) = \pi_{i=1]^n P(x_i | x_{i-1}, ..., x_n)
        """
        backprobs = map(lambda rv: prob(rv, given=rv.parents()), rvs)
        product(backprobs)
    def get(self, abbr):
        return self.abbrs[abbr]
    def name(self, name):
        return self.nodes[name]
    def query(self, string):
        """
           returns  Pr( rv | given )
        """
        rvs, given = self.parse_query(string)
        return 

    def parse_query(self, query):
        """ maps string query -> rv, given list"""
        rvs, given = tuple(map(lambda x: x.split(','), query.split('|')))
        return (rvs, given)

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
    print("Made BayesNet:\n")
#    print(bn)
    print(bn.nodes)
#    bn.query_raw(rv=[b], given=[j, m]) #probability of a burglary given both john and mary call
#    bn.Pr('B | j, m')

