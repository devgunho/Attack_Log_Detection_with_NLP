import tensorflow as tf

tf.test.is_gpu_available()


import numpy as np
import pandas as pd


import os


class Config:
    PROJECT_DIR = os.environ["PWD"]
    DATA_DIR = os.getenv("DATA_DIR", "dat/")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "results/")
    MODELS_DIR = os.getenv("MODELS_DIR", "models/")
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "models/checkpoint/")
    LOGS_DIR = os.getenv("LOGS_DIR", "logs/")


TRUE_TARGET_FILE = "../../clean_anomaly_data_sample/HDFS_Anomaly_10k.log_clean.txt"
FALSE_TARGET_FILE = "../../clean_anomaly_data_sample/HDFS_Anomaly_10k.log_clean.txt"


def print_stats(X, y, num_s, num_e, ratio):
    print("Stats:")
    print("------------------------")
    print("------------------------")
    print(f"N(X) == N(y) == {len(y)}")
    print(f"errs: {num_e}")
    print(f"Clean data (N = {num_s}) ratio: {ratio}%")
    print("------------------------")


config = Config()

X_train_dir = f"{config.DATA_DIR}clean/"
y_train_dir = f"{config.DATA_DIR}trans/"
X_test_dir = f"{config.DATA_DIR}test/"

# Will notify if these values change
max_encoder_seq_length = 81
max_decoder_seq_length = 162


import string

table = str.maketrans(dict.fromkeys(string.punctuation))

# All of the characters and substring that would mark lines in the training data as "faulty"
invalid_chars = set(
    [
        ":",
        "+",
        "#",
        "@",
        "Ö",
        "á",
        "ä",
        "é",
        "í",
        "ñ",
        "ó",
        "ö",
        "ú",
        "ā",
        "Ć",
        "ć",
        "ʻ",
        "́",
        "е",
        "н",
        "о",
        "п",
        "у",
        "ш",
        "ã",
        "ï",
        "ō",
        "ū",
        "ί",
        "α",
        "δ",
        "ε",
        "κ",
        "ο",
        "в",
        "ὐ",
        chr(776),
        "ç",
        "ē",
        "D",
        "O",
        "T",
    ]
)
invalid_chars_X = set(["(", ")", "<", ">", "_", ","])
invalid_markers = set(["\\F", "TrueP", "\\x", "semantics_error", "Prog("])
files_with_compound_preds = [20, 21, 15]


def mark_if_faulty(line, file_idx, X=False):
    if X and (
        any((c in invalid_chars) for c in line)
        or any((c in invalid_chars_X) for c in line)
    ):
        return "syntax_error"
    # TODO: Refactor this hacky workaround
    if line[0] == "(" and file_idx not in files_with_compound_preds:
        return "syntax_error"
    if any((m in line) for m in invalid_markers) or any(
        (c in invalid_chars) for c in line
    ):
        return "syntax_error"
    # Remove top-level parentheses from lambda expression
    if line[0:4] == "(exi" and line[-1] == ")":
        line = line[1:-1]
    if line[0:4] == "(all" and line[-1] == ")":
        line = line[1:-1]

    return line


def lines_from_file(direc, name, drop_punc=False, lower=True, drop_fullstop=True):
    with open(direc + name) as f:
        for l in f:
            l = l.rstrip()
            if drop_punc:
                l = l.translate(table)
            if lower:
                l = l.lower()
            if drop_fullstop and not drop_punc:
                l = l[0:-1]
            yield l


def load_and_clean_data(start_idx=1, end_idx=17, skip_idx_list=None):
    X, y = [], []

    err = lambda x: x == "syntax_error"
    X_name = lambda i: f"concordance_{i}_clean.txt"
    y_name = lambda i: f"concordance_{i}_clean.lam"

    # Load lines from files and mark those that are "faulty"
    for i in range(start_idx, end_idx + 1):
        if i in skip_idx_list:
            continue

        X = X + [
            mark_if_faulty(line, i, True)
            for line in lines_from_file(X_train_dir, X_name(i), drop_fullstop=True)
        ]
        y = y + [
            mark_if_faulty(line, i)
            for line in lines_from_file(
                y_train_dir, y_name(i), lower=False, drop_fullstop=False
            )
        ]

    # Save "faulty" line indices
    err_idx_X = [i1 for i1 in range(len(X)) if err(X[i1])]
    err_idx_y = [j1 for j1 in range(len(X)) if err(y[j1])]

    err_idx = set(err_idx_X).union(set(err_idx_y))
    num_err = len(err_idx)
    num_samples = len(y) - num_err
    clean_ratio = 100 - ((num_err / len(y)) * 100)

    # Show stats about training data
    print_stats(X, y, num_samples, num_err, clean_ratio)

    # Remove "faulty" lines
    for index in sorted(list(err_idx), reverse=True):
        del X[index]
        del y[index]

    return (X, y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    *load_and_clean_data(1, 20, []), test_size=0.25, random_state=4, shuffle=True
)


# Vectorize the data.
input_characters = set()
target_characters = set()

for i in range(0, len(X_train)):
    # SOS == '\n'
    # EOS == '\t'
    y_train[i] = "\t" + y_train[i] + "\n"

    for char in X_train[i]:
        if char not in input_characters:
            input_characters.add(char)
    for char in y_train[i]:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_X_len = max([len(txt) for txt in X_train])
max_y_len = max([len(txt) for txt in y_train])

print("Number of samples:", len(X_train))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)

if max_X_len > max_encoder_seq_length:
    print("WARNING: NEW Max sequence length for inputs:", max_X_len)
    print("Dataset may be incompatible with older models.")
    max_encoder_seq_length = max_X_len

if max_y_len > max_decoder_seq_length:
    print("WARNING: NEW Max sequence length for outputs:", max_y_len)
    print("Dataset may be incompatible with older models.")
    max_decoder_seq_length = max_y_len

print(target_characters)


# char to index
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(X_train), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(X_train), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(X_train), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)


import sys

np.set_printoptions(threshold=10)

for i, (input_text, target_text) in enumerate(zip(X_train, y_train)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

encoder_input_data[0]


import os
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(
            name="W_a",
            shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
            initializer="uniform",
            trainable=True,
        )
        self.U_a = self.add_weight(
            name="U_a",
            shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
            initializer="uniform",
            trainable=True,
        )
        self.V_a = self.add_weight(
            name="V_a",
            shape=tf.TensorShape((input_shape[0][2], 1)),
            initializer="uniform",
            trainable=True,
        )

        super(AttentionLayer, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print("encoder_out_seq>", encoder_out_seq.shape)
            print("decoder_out_seq>", decoder_out_seq.shape)

        def energy_step(inputs, states):
            """Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            assert_msg = "States must be an iterable. Got {} of type {}".format(
                states, type(states)
            )
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(
                K.dot(inputs, self.U_a), 1
            )  # <= batch_size, 1, latent_dim
            if verbose:
                print("Ua.h>", U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print("Ws+Uh>", Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print("ei>", e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """Step function for computing ci using ei"""

            assert_msg = "States must be an iterable. Got {} of type {}".format(
                states, type(states)
            )
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print("ci>", c_i.shape)
            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(
            encoder_out_seq, axis=2
        )  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step,
            decoder_out_seq,
            [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step,
            e_outputs,
            [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """Outputs produced by the layer"""
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1])),
        ]


from tensorflow.keras.layers import (
    Input,
    LSTM,
    GRU,
    Dense,
    Concatenate,
    TimeDistributed,
    Bidirectional,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from livelossplot import PlotLossesKeras


# ### Bidirectional LSTM + Attention
#

latent_dim = 44
batch_size = 48  # Batch size for training.
epochs = 30  # Number of epochs to train for.
recurrent_dropout_rate = 0.2
simple_name = "bilstm"

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
decoder_inputs = Input(shape=(None, num_decoder_tokens))

# Encoder LSTM
encoder_lstm = Bidirectional(
    LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm",
        recurrent_dropout=recurrent_dropout_rate,
    ),
    name="bidirectional_encoder",
)
(
    encoder_out,
    encoder_fwd_state_h,
    encoder_fwd_state_c,
    encoder_back_state_h,
    encoder_back_state_c,
) = encoder_lstm(encoder_inputs)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(
    latent_dim * 2,
    return_sequences=True,
    return_state=True,
    recurrent_dropout=recurrent_dropout_rate,
    name="decoder_lstm",
)
decoder_out, _, _ = decoder_lstm(
    decoder_inputs,
    initial_state=[
        Concatenate(axis=-1)([encoder_fwd_state_h, encoder_back_state_h]),
        Concatenate(axis=-1)([encoder_fwd_state_c, encoder_back_state_c]),
    ],
)

# Attention layer
attn_layer = AttentionLayer(name="attention_layer")
attn_out, attn_states = attn_layer([encoder_out, decoder_out])

# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name="concat_layer")(
    [decoder_out, attn_out]
)

# Dense layer
dense = Dense(num_decoder_tokens, activation="softmax", name="softmax_layer")
decoder_pred = dense(decoder_concat_input)


opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.0015,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=True,
    name="RMSprop",
)

# Full model
full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
full_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])


full_model.summary()


# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence_bigru_attn(
    encoder_model, decoder_model, test_X_seq, num_encoder_tokens, num_decoder_tokens
):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_X_seq: sequence of word ids
    :param num_encoder_tokens: int
    :param num_decoder_tokens: int
    :return:
    """

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    enc_outs, enc_fwd_state, enc_back_state = encoder_model.predict(test_X_seq)
    dec_state = np.concatenate([enc_fwd_state, enc_back_state], axis=-1)

    attention_weights = []
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        dec_out, attention, dec_state = decoder_model.predict(
            [enc_outs, dec_state, target_seq]
        )

        # Sample a token
        sampled_token_index = np.argmax(dec_out, axis=-1)[0, 0]
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        attention_weights.append((sampled_token_index, attention))

    return decoded_sentence, attention_weights


def decode_sequence_bilstm_attn(
    encoder_model, decoder_model, test_X_seq, num_encoder_tokens, num_decoder_tokens
):
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    (
        enc_outs,
        enc_fwd_state_h,
        enc_fwd_state_c,
        enc_back_state_h,
        enc_back_state_c,
    ) = encoder_model.predict(test_X_seq)
    encoder_state_h = np.concatenate([enc_fwd_state_h, enc_back_state_h], axis=-1)
    encoder_state_c = np.concatenate([enc_fwd_state_c, enc_back_state_c], axis=-1)

    # The ordering seems significant
    # enc_outs, enc_fwd_state_h, enc_fwd_state_c, enc_back_state_h, enc_back_state_c = encoder_model.predict(test_X_seq)

    attention_weights = []
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        dec_out, attention, dec_state_h, dec_state_c = decoder_model.predict(
            [enc_outs, encoder_state_h, encoder_state_c, target_seq]
        )

        # Sample a token
        sampled_token_index = np.argmax(dec_out[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        attention_weights.append((sampled_token_index, attention))

        # Update states
        encoder_state_h = dec_state_h
        encoder_state_c = dec_state_c

    return decoded_sentence, attention_weights


def decode_sequence_lstm(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]

    return decoded_sentence


latent_dim = 44

full_model.load_weights(
    f"{config.CHECKPOINT_DIR}weights-{simple_name}-N({len(X_train)})-{latent_dim}.best.hdf5"
)
loaded_model = full_model


""" Encoder (Inference) model """
encoder_inf_inputs = Input(shape=(None, num_encoder_tokens), name="encoder_inf_inputs")
(
    encoder_inf_out,
    encoder_inf_fwd_state_h,
    encoder_inf_fwd_state_c,
    encoder_inf_back_state_h,
    encoder_inf_back_state_c,
) = encoder_lstm(encoder_inf_inputs)
encoder_model = Model(
    inputs=encoder_inf_inputs,
    outputs=[
        encoder_inf_out,
        encoder_inf_fwd_state_h,
        encoder_inf_fwd_state_c,
        encoder_inf_back_state_h,
        encoder_inf_back_state_c,
    ],
)

""" Decoder (Inference) model """
decoder_inf_inputs = Input(shape=(None, num_decoder_tokens), name="decoder_inf_inputs")
encoder_inf_states = Input(
    batch_shape=(None, None, 2 * latent_dim), name="encoder_inf_states"
)

decoder_state_input_h = Input(batch_shape=(None, 2 * latent_dim))
decoder_state_input_c = Input(batch_shape=(None, 2 * latent_dim))

decoder_init_state = [decoder_state_input_h, decoder_state_input_c]

decoder_inf_out, decoder_inf_state_h, decoder_inf_state_c = decoder_lstm(
    decoder_inf_inputs, initial_state=decoder_init_state
)

attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
decoder_inf_concat = Concatenate(axis=-1, name="concat")(
    [decoder_inf_out, attn_inf_out]
)
decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)

decoder_model = Model(
    inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
    outputs=[
        decoder_inf_pred,
        attn_inf_states,
        decoder_inf_state_h,
        decoder_inf_state_c,
    ],
)


import sys


def test_true_func(line):
    X_test = [line]

    encoder_test_data = np.zeros(
        (len(X_test), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    for i, x in enumerate(X_test):
        for t, char in enumerate(x):
            encoder_test_data[i, t, input_token_index[char]] = 1.0
        encoder_test_data[i, t + 1 :, input_token_index[" "]] = 1.0

    correct = 0
    checked = 0
    dec_line = ""
    for seq_index in range(0, len(X_test)):
        test_X = X_test[seq_index]
        input_seq = encoder_test_data[seq_index : seq_index + 1]

        # Bi-LSTM
        decoded_sentence, attn_weights = decode_sequence_bilstm_attn(
            encoder_model,
            decoder_model,
            input_seq,
            num_encoder_tokens,
            num_decoder_tokens,
        )

        print("-")
        origin_input = X_test[seq_index]
        print("Input sentence:", origin_input)
        print("Decoded sentence:", repr(decoded_sentence.rstrip()))
        dec_line = repr(decoded_sentence.rstrip())
        w1 = dec_line.split("_")[1].split("(")[0]
        w2 = dec_line.split("_")[2].split("(")[0]
        if (
            origin_input.find(" " + w1 + " ") != -1
            or origin_input.find(" " + w2 + " ") != -1
        ):
            # print(origin_input)
            return True, dec_line
        else:
            # print(origin_input)
            return False, dec_line


def true_test():
    file = open(TRUE_TARGET_FILE, "r")
    lines = file.readlines()
    result_file = open("result_true.txt", "w")

    true_cnt = 0
    false_cnt = 0
    error_cnt = 0
    line_cnt = 0

    for line in lines:
        line_cnt += 1
        print("Line No. :", line_cnt)
        target_line = line.strip()
        func_res = test_false_func(target_line)

        if func_res[0] == "ERROR":
            error_cnt += 1
            result_file.write(f"ERROR " + str(error_cnt) + " : ")
            result_file.write(target_line + "  >>>  " + str(func_res) + "\n")

        elif func_res[0]:
            true_cnt += 1
            result_file.write(f"TRUE " + str(true_cnt) + " : ")
            result_file.write(target_line + "  >>>  " + str(func_res) + "\n")

        else:
            false_cnt += 1
            result_file.write(f"FALSE " + str(false_cnt) + " : ")
            result_file.write(target_line + "  >>>  " + str(func_res) + "\n")

    result_file.write("\n---\n" + "TRUE: " + str(true_cnt))
    result_file.write("FALSE: " + str(false_cnt))
    result_file.write("ERROR: " + str(error_cnt))
    result_file.close()


def test_false_func(line):
    X_test = [line]

    encoder_test_data = np.zeros(
        (len(X_test), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    try:
        for i, x in enumerate(X_test):
            for t, char in enumerate(x):
                encoder_test_data[i, t, input_token_index[char]] = 1.0
            encoder_test_data[i, t + 1 :, input_token_index[" "]] = 1.0
    except:
        return "ERROR", ""

    correct = 0
    checked = 0
    dec_line = ""
    for seq_index in range(0, len(X_test)):
        test_X = X_test[seq_index]
        input_seq = encoder_test_data[seq_index : seq_index + 1]

        # Bi-LSTM
        decoded_sentence, attn_weights = decode_sequence_bilstm_attn(
            encoder_model,
            decoder_model,
            input_seq,
            num_encoder_tokens,
            num_decoder_tokens,
        )

        print("-")
        origin_input = X_test[seq_index]
        print("Input sentence:", origin_input)
        print("Decoded sentence:", repr(decoded_sentence.rstrip()))
        dec_line = repr(decoded_sentence.rstrip())
        w1 = dec_line.split("_")[1].split("(")[0]
        w2 = dec_line.split("_")[2].split("(")[0]
        if (
            origin_input.find(" " + w1 + " ") != -1
            and origin_input.find(" " + w2 + " ") != -1
        ):
            # print(origin_input)
            return True, dec_line
        else:
            # print(origin_input)
            return False, dec_line


def false_test():
    file = open(FALSE_TARGET_FILE, "r")
    lines = file.readlines()
    result_file = open("result_false.txt", "w")

    true_cnt = 0
    false_cnt = 0
    error_cnt = 0
    line_cnt = 0

    for line in lines:
        line_cnt += 1
        print("Line No. :", line_cnt)
        target_line = line.strip()
        func_res = test_false_func(target_line)

        if func_res[0] == "ERROR":
            error_cnt += 1
            result_file.write(f"ERROR " + str(error_cnt) + " : ")
            result_file.write(target_line + "  >>>  " + str(func_res) + "\n")

        elif func_res[0]:
            true_cnt += 1
            result_file.write(f"TRUE " + str(true_cnt) + " : ")
            result_file.write(target_line + "  >>>  " + str(func_res) + "\n")

        else:
            false_cnt += 1
            result_file.write(f"FALSE " + str(false_cnt) + " : ")
            result_file.write(target_line + "  >>>  " + str(func_res) + "\n")

    result_file.write("\n---\n" + "TRUE: " + str(true_cnt) + "\n")
    result_file.write("FALSE: " + str(false_cnt) + "\n")
    result_file.write("ERROR: " + str(error_cnt) + "\n")
    result_file.close()


true_test()
false_test()
