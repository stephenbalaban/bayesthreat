import functools
import itertools
from math import log

it = itertools
ft = functools

DEBUG = False
NEGCHAR = '!'
SEPCHAR = ','
PIPCHAR = '|'

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

#
# utilities
#

mult = lambda a,b: a * b
product = lambda l: functools.reduce(mult, l)
identity = lambda x: x
pr_not = lambda pr: 1 - pr

def flatten(l):
    """
    Takes a nested list of lists (two deep) and flattens it into a single list:
    [[1,2],[3,4],[5,6]] -> [1,2,3,4,5,6]
    """
    return [item for sublist in l for item in sublist]

##
# Data Structure:
#
#   Node
#       A node is a 4-tuple [name, cpt, abbr, parents]
#   Bayesnet
#       A Bayesnet is a dict of nodes keyed by their abbreviation
#           'a': node_a
#           'b': node_b
##

#
# Node functions
#

def node(name, abbr, cpt, parents=[]):
    """
    A node is a 4-tuple:
        name, constraint_pr_table, abbreviation, parents_list
    e.g.
        >>> bn.node('Female', 'f', { (): 0.5}, []) 
        ['Female', {(): 0.5}, 'f', []]
    """
    return [name, cpt, abbr, parents]

def node_abbr_not(abbr, negchar=NEGCHAR):
    """ 
    Returns the opposite abbr of a given node
    e.g.
        >>> node_abbr_not('nodename')
        '!nodename'
        >>> node_abbr_not('!nodename')
        'nodename'
    """
    if negchar in abbr:
        return abbr.split(negchar)[0]
    else:
        return '!%s' % abbr

def parents(node):
    return node[3]

def abbr(node):
    return node[2]

def name(node):
    return node[0]

def cpt(node):
    return node[1]

def keys(node):
    """
    node -> cpt keys
    """
    return list(cpt(node).keys())

def node_pr(node, given_key=(), is_pos=True):
    """ 
    (node, given_list, is_positive) -> float
    """
    wrapper = identity if is_pos else pr_not
    if given_key not in cpt(node):
        raise KeyError("Key for cpt(node) not provided, got: %s" % str(given_key))
    else:
        return wrapper(cpt(node)[given_key])

def parse_sep(sepstr, sepchar=SEPCHAR):
    """
    >>> parse_sep('a,b,c')
    ['a','b','c']
    >>> parse_sep(',a,b,c') # no empty strings
    ['a','b','c']
    """
    return list(filter(bool, [x.strip() for x in sepstr.split(sepchar)]))

def negate(rvs, negchar=NEGCHAR):
    """
    rv_list -> neg_rv_list

    >>> negate(['a', '!b', 'c'])
    ['!a', 'b', '!c']
    >>> negate(['a'])
    ['a']
    """
    return list(map(lambda x: x.split(negchar)[-1] if negchar in x else 
                                                        '!%s' % x, rvs))

def parse_query(st, pipechar=PIPCHAR, sepchar=SEPCHAR, do_negate=False):
    """
    query_str -> rvs, given
    >>> parse_query('b | m,j')
    ('b', ('m','j')) 
    """
    if pipechar in st:
        dirty_rvs, dirty_given = st.split(pipechar)
        given = parse_sep(dirty_given)
    else:
        dirty_rvs = st
        given = []
    rvs = parse_sep(dirty_rvs)
    if do_negate:
        rvs = negate(rvs)
    return rvs, given

def choose_key(query_list):
    def choose(key):
        return all([abbr in query_list for abbr in key])
    return choose

def first(l):
    l = tuple(l)
    if len(l):
        return l[0]
    return l

def given_key(node, query, negchar=NEGCHAR):
    """
    (node, query_str) -> cpt_key
    e.g.
    >>> (e['m'], 'e, b, a, m, j') -> 'a'
    >>> (e['m'], 'e, b, !a, m, j') -> '!a'
    >>> cpt(e['m'])
    {'a': 0.7, '!a': 0.01}
    """
    return first(filter(choose_key(query), keys(node)))

#
#   Bayes Net Functions
#

def bnet(*bnodes):
    """
    list_of_bnodes -> dict_of_bnodes 
    """
    return dict([(abbr(node), node) for node in bnodes])

def bnet_nodes(bnet):
    return bnet.values()

def key_set(bnet):
    return set(bnet.keys())

def is_positive(node, query, negchar=NEGCHAR):
    """
    (node, query_str) -> Bool
    e.g.
        >>> ('e', 'j,m,a,!b,!e')
        False
    """
    f = list(filter(lambda abbrv: abbr(node) in abbrv, query))
    if len(f):
        return negchar not in f[0]
    return False

def unique(tup, negchar=NEGCHAR):
    """
    predicate for whether or not a tuple contains the same abbr

    e.g.
        >>> bn.unique(('a', '!a')) 
        False
        >>> bn.unique(('a', '!b'))
        True
    """
    return len(set(map(lambda x: x.split(negchar)[-1], tup))) == \
           len(list(map(lambda x: x.split(negchar)[-1], tup)))

def remove_alike(abbr_list, negchar=NEGCHAR):
    """
    removes tuples like ('a', '!a'),
                        ('c', '!a', '!c')
    e.g.
        >>> bn.remove_alike([('a', '!a'), ('b', 'c')]) 
        [('b', 'c')]
    """
    return list(filter(unique, abbr_list))

def combinate(ndlist):
    """
    takes a list of thingies and returns all possible permutations (with !) of
    those variables

    e.g.
        >>> bn.combinate(['a', 'b'])
        [('!a', '!b'), ('!a', 'b'), ('!b', 'a'), ('a', 'b')]
    """
    return remove_alike(it.combinations(
            ([node_abbr_not(node) for node in ndlist] + list(ndlist)),
            len(ndlist)))

def remove_neg(n, negchar=NEGCHAR):
    """
    rv_abbr -> positive_rv_abbr
        >>> bn.remove_neg('!a')
        'a'
        >>> bn.remove_neg('a')
        'a'
    """
    if negchar in n:
        return n.split(negchar)[-1]
    else:
        return n

def hidden_vars(bnet, rv, given):
    """
    (bnet, rv, given) -> hidden_vars
    """
    return key_set(bnet) - set(map(remove_neg, rv + given))

def alpha(a, b):
    """
    alpha is a normalization constant
    """
    return 1 / (a + b)

def normalize(a, b):
    """
    pr(A), pr(B) -> alpha(a,b)*pr(A)

    e.g.
        >>> a = 0.03
        >>> b = 0.0004
        >>> a_norm = bn.normalize(a, b)
        >>> b_norm = bn.normalize(b, a)
        >>> print(a_norm, b_norm) 
        0.9868421052631577, 0.013157894736842105
    """
    return alpha(a, b) * a

def generate_queries(bnet, parsed_query):
    """
    bnet, query -> [query]

    # debugging lol
    """
    rv, given = parsed_query
    combos = combinate(hidden_vars(bnet, rv, given))
    return [set(x).union(rv).union(given) for x in combos]

def calc_pr(bnet, query):
    """
    (bnet, query) -> float
    """
    return product([node_pr(node, given_key=given_key(node, query),
                                  is_pos=is_positive(node, query))
                    for node in bnet_nodes(bnet)])

def pr(bnet, query_string):
    return normalize(sum([calc_pr(bnet, query) for query in
                          generate_queries(bnet, parse_query(query_string))]),
                     sum([calc_pr(bnet, query) for query in 
                          generate_queries(bnet, parse_query(query_string, do_negate=True))]))
        

#
# Models
#

def make_eq_model():
    """
    Generates the basic burglary & earthquake model from 
    Artificial Intelligence: A Modern Approach
    """
    ecpt = { (): 0.002 }
    bcpt = { (): 0.001 }
    acpt = { ('b', 'e'): 0.95,
             ('b', '!e'): 0.94,
             ('!b', 'e'): 0.29,
             ('!b', '!e'): 0.001 }
    mcpt = { ('a',): 0.70,
             ('!a',): 0.01 }
    jcpt = { ('a',): 0.90,
             ('!a',): 0.05 }
    e = node('Earthquake', 'e', ecpt, parents=[])
    b = node('Burgler',    'b', bcpt, parents=[])
    a = node('Alarm',      'a', acpt, parents=[e,b])
    m = node('Mary',       'm', mcpt, parents=[a])
    j = node('John',       'j', jcpt, parents=[a])
    bn = bnet(e,b,a,m,j)
    return bn

def make_writer_model():
    """
    resume    writing sample
    |               |
    +-> R       W <-+
         \     /
          \   /
            A <- accepted
            |
            J <- accept job
    """
    rcpt = { (): 0.25 }
    wcpt = { (): 0.3 }
    acpt = { ('r', 'w'): 0.95,
             ('r', '!w'): 0.8,
             ('!r', 'w'): 0.25,
             ('!r', '!w'): 0.03 }
    jcpt = { ('a',): 0.4,
             ('!a',): 0 }
    r = node('GoodResume', 'r', rcpt, parents=[])
    w = node('GoodWritingSample', 'w', wcpt, parents=[])
    a = node('Accepted', 'a', acpt, parents=[r,w])
    j = node('TakesJob', 'j', jcpt, parents=[a])
    bn = bnet(r,w,a,j)
    return bn

def run():
    make_eq_model()

if __name__ == '__main__':
    run()
