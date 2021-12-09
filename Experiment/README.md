## Neural Machine Translation (seq2seq): English → Predicate Logic

Predicate logic, first-order logic or quantified logic is a formal language in which propositions are expressed in terms of predicates, variables and quantifiers.
It is different from propositional logic which lacks quantifiers.

It should be viewed as an extension to propositional logic, in which the notions of truth values, logical connectives, etc still apply but propositional letters(which used to be atomic elements), will be replaced by a newer notion of proposition involving predicates and quantifiers.

```
Input sentence: every cat likes fish
Decoded sentence: 'all x1.(_cat(x1) -> exists x2.(_fish(x2) & _like(x1,x2)))'

Input sentence: some people are evil and some people are good
Decoded sentence: 'exists x1.(_people(x1) & _good(x1)) & exists x2.(_people(x2) & _evil(x2))'

Input sentence: she is willing and able
Decoded sentence: 'exists x1.(_able(x1) & _willing(x1))'
```

<br/>

### Models

1. Plain LSTM
2. Plain GRU
3. Bi-directional GRU + Attention (Bahdanau, 2014)
4. Bi-directional LSTM + Attention

<br/>

## Experiment Env.

```
conda
```
