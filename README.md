# Bayes Threat - Bayesian Networks, code tutorial
Stephen A. Balaban
2012-03-10

The purpose of this code tutorial is to act as a playground for me to learn
more about Bayesian networks as well as a roadmap for others who are interested
in diving into the rich world of probabilistic inference. Most of the content
within this tutorial is condensed knowledge from <a
href="http://aima.cs.berkely.edu/index.html">Russell & Norvig's Artificial
Intelligence: A Modern Approach</a>. The book is a wealth of knowledge
regarding Artificial Intelligence, I recommend it as part of anyone's <a
href="/ppl" alt="Personal Professional Library">PPL</a>.

## What is a Bayesian Network

Simply a bayesian network is a directed acyclic 

## The Structure

<code>bn.py</code> contains two main sets of functions, node-level functions and
bayesnet-level functions

The two constructor functions, *node* and *bnet*,
create nodes and Bayesian networks respectively.

In my model, a *node* is a 4-tuple: 
    name, constraint_probability_table, abbreviation, parent_list

A *bnet* is a list of nodes in the network. Let's construct the model from
AIMA:

<pre>
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
</pre>
