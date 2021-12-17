for i in range(0, len(X_train)):
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


if max_X_len > max_encoder_seq_length:
    max_encoder_seq_length = max_X_len

if max_y_len > max_decoder_seq_length:
    max_decoder_seq_length = max_y_len

print(target_characters)

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