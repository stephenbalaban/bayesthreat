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
    | 0 | 0.05 |               | 1 | 0.00 |
    +---+------+               +---+------+

    P(x_1, ..., x_n) = \pi_{i=1]^n P(x_i | x_{i-1}, ..., x_n)
    Chain rule:
                     = \pi_{i=1]^n P(x_i | Parents(X_i))

BN with continuous variables:

    Not all problems can be described using boolean nodes. For example, if the
    amount of money spent on a financial transaction increases the probability
    of a fraud occurring, we might want to model the transaction amount as
    a continuous variable.

"""

class RV(object):
    def __init__(self, name, pdist):
        self.name = name
        self.pdist = make_pdist(pdist)
    def make_pdist(self, pdist):
        result = {}
        for k in pdist:
            result[k] = pdist[k]
        if len(pdist) == 1: # binary RV
            result[not pdist[0]] = 1 - pdist[0]
        return result

class BayesCPT(object):
    def __init__(self):
        pass

class BayesNode(object):
    def __init__(self, name, cpt, parent_list=[]):
        self.name = name
        self.cpt = cpt
        self.parents = parent_list

class BayesNet(object):
    def __init__(self, nodes):
        self.nodes = nodes

def make_earthquake_model():
    erv = RV("Earthquake", {True: 0.002})
    brv = RV("Burgler", {True: 0.001})
    arv = RV("Alarm", {True: 0.002})
    erv = RV("MaryCalls", {True: 0.002})
    erv = RV("JohnCalls", {True: 0.002})
    e = BayesNode(erv, e_cpt)
    b = BayesNode(brv, b_cpt)
    a = BayesNode(arv, a_cpt, [e, b])
    j = BayesNode(jrv, j_cpt, [a])
    m = BayesNode(mrv, m_cpt, [a])

