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
# Make python packagelist.txt
conda list --export > packagelist.txt

# Using packagelist.txt file
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: osx-arm64
```

<br/>

<br/>

## Model Description

```
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_19 (InputLayer)          [(None, None, 37)]   0           []                               
                                                                                                  
 bidirectional_encoder (Bidirec  [(None, None, 88),  28864       ['input_19[0][0]']               
 tional)                         (None, 44),                                                      
                                 (None, 44),                                                      
                                 (None, 44),                                                      
                                 (None, 44)]                                                      
                                                                                                  
 input_21 (InputLayer)          [(None, None, 45)]   0           []                               
                                                                                                  
 concatenate_6 (Concatenate)    (None, 88)           0           ['bidirectional_encoder[0][1]',  
                                                                  'bidirectional_encoder[0][3]']  
                                                                                                  
 concatenate_7 (Concatenate)    (None, 88)           0           ['bidirectional_encoder[0][2]',  
                                                                  'bidirectional_encoder[0][4]']  
                                                                                                  
 decoder_lstm (LSTM)            [(None, None, 88),   47168       ['input_21[0][0]',               
                                 (None, 88),                      'concatenate_6[0][0]',          
                                 (None, 88)]                      'concatenate_7[0][0]']          
                                                                                                  
 attention_layer (AttentionLaye  ((None, None, 88),  15576       ['bidirectional_encoder[0][0]',  
...
Total params: 99,573
Trainable params: 99,573
Non-trainable params: 0
```

<br/>

## Windows

### Data Info.

```
Stats:
------------------------
------------------------
N(X) == N(y) == 40000
errs: 0
Clean data (N = 40000) ratio: 100.0%
------------------------
```

### Train

```
accuracy
	training         	 (min:    0.901, max:    0.989, cur:    0.989)
	validation       	 (min:    0.984, max:    0.989, cur:    0.989)
Loss
	training         	 (min:    0.026, max:    0.366, cur:    0.026)
	validation       	 (min:    0.026, max:    0.061, cur:    0.026)

Epoch 28: val_loss did not improve from 0.02627
563/563 [==============================] - 532s 945ms/step - loss: 0.0264 - accuracy: 0.9887 - val_loss: 0.0263 - val_accuracy: 0.9886
Epoch 28: early stopping
```

![image-20221213195256942](./README.assets/win-train.png)

### Recognition

```
# windows 2000
TP : 1990
FN : 10

# linux 2000 + hadoop 2000 + hdfs 2000 + spark 2000
TN : 7981
FP : 19
```

<br/>

<br/>

## Linux

### Data Info.

```
Stats:
------------------------
------------------------
N(X) == N(y) == 40000
errs: 0
Clean data (N = 40000) ratio: 100.0%
------------------------
```

### Train

```
accuracy
	training         	 (min:    0.913, max:    0.987, cur:    0.987)
	validation       	 (min:    0.975, max:    0.987, cur:    0.987)
Loss
	training         	 (min:    0.033, max:    0.333, cur:    0.033)
	validation       	 (min:    0.032, max:    0.093, cur:    0.032)

Epoch 30: val_loss did not improve from 0.03206
563/563 [==============================] - 308s 548ms/step - loss: 0.0328 - accuracy: 0.9873 - val_loss: 0.0322 - val_accuracy: 0.9873
```

![image-20221214090220787](./README.assets/linux-train.png)

### Recognition

```
# linux 2000
TP :
FN : 

# windows 2000 + hadoop 2000 + hdfs 2000 + spark 2000
TN :  
FP : 
```

<br/>

<br/>

## Hadoop

### Data Info.

```

```

### Train

```
accuracy
	training         	 (min:    0.891, max:    0.983, cur:    0.983)
	validation       	 (min:    0.964, max:    0.983, cur:    0.983)
Loss
	training         	 (min:    0.046, max:    0.402, cur:    0.046)
	validation       	 (min:    0.046, max:    0.128, cur:    0.046)

Epoch 30: val_loss did not improve from 0.04569
563/563 [==============================] - 4105s 7s/step - loss: 0.0463 - accuracy: 0.9831 - val_loss: 0.0457 - val_accuracy: 0.9832
```

![image-20221220194946017](./README.assets/hadoop-train.png)

### Recognition

```
# hadoop 2000
TP : 1724
FN : 276

# windows 2000 + linux 2000 + hdfs 2000 + spark 2000
TN : 8000
FP : 0
```

### Detection

```
# anomaly 4209 (10000)
TP : 
FN : 
TN : 
FP : 
```

<br/>

<br/>

## HDFS

### Data Info.

```

```

### Model Info.

```

```

### Train

```

```

### Recognition

```
# hdfs 2000
TP : 1998
FN : 2

# windows 2000 + linux 2000 + hadoop 2000 + spark 2000
TN : 7999 
FP : 1
```

### Detection

```
# anomaly 423 (10000)
TP : 
FN :
TN :
FP :
```

<br/>

<br/>

