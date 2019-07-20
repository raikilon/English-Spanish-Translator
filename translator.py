import tensorflow as tf
import unicodedata
import numpy as np
import matplotlib as mplt

mplt.use('Agg')
import matplotlib.pyplot as plt
import os
import re

# 1.1.1.7
SOS = "GO"
EOS = "EOS"
PAD = "PAD"

# create session
session = tf.Session()


def read_and_prepare_data(number_of_sentences, output=False):

    # random number to print sentence to check preprocessing
    random_num = int(np.random.rand(1) * number_of_sentences)

    # 1.1.1.1
    # Retrieve data for training
    path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip',
                                          extract=True)
    path_to_file = os.path.dirname(path_to_zip) + '/spa-eng/spa.txt'

    # open file
    f = open(path_to_file, 'r', encoding='UTF-8')

    # List which contain the spanish and english sentences
    eng_sentences = []
    spa_sentences = []

    # 1.1.1.3
    # Go through text file and split row to get spanish and english sentences
    for x in range(number_of_sentences):
        temp = next(f).split("\t")
        eng_sentences.append(temp[0])
        spa_sentences.append(temp[1])

    # print random sentence
    if output:
        print("This is a sentence '{0}'".format(eng_sentences[random_num]))

    # 1.1.1.10
    # preprocess eng sentences for target value (they do not contain the SOS value)
    eng_sentences_target = preprocess_sequences(eng_sentences, True)
    # normal preprocessed sentences
    eng_sentences = preprocess_sequences(eng_sentences)
    spa_sentences = preprocess_sequences(spa_sentences)

    # create dictionary for both languages
    eng_voc = create_vocabulary(eng_sentences)
    spa_voc = create_vocabulary(spa_sentences)
    # generate some statistical values and print them
    max_len_eng, max_len_spa = perform_statistics(eng_voc, spa_voc, eng_sentences,
                                                  spa_sentences, True)
    # transform sentences to int values to feed to the network
    eng_int = word2int(eng_voc, eng_sentences)
    target_int = word2int(eng_voc, eng_sentences_target)
    spa_int = word2int(spa_voc, spa_sentences)

    # print random sentence to check preprosessing
    if output:
        print("This is a preprocessed sentence '{0}'".format(eng_sentences[random_num]))
        print("This is a its translation to int '{0}'".format(eng_int[random_num]))

    # 1.1.2.1
    # compute sentences lengths before padding them (this is useful for the loss computation)
    eng_sizes = sentences_sizes(eng_sentences)

    # 1.1.2.2
    # pad sentences to language max length
    eng_padded = tf.keras.preprocessing.sequence.pad_sequences(eng_int, maxlen=max_len_eng, padding='post')
    target = tf.keras.preprocessing.sequence.pad_sequences(target_int, maxlen=max_len_eng, padding='post')
    spa_padded = tf.keras.preprocessing.sequence.pad_sequences(spa_int, maxlen=max_len_spa, padding='post')

    return eng_voc, spa_voc, max_len_eng, max_len_spa, eng_sizes, eng_padded, spa_padded, target


def create_data_set(partitions, total_num_sentences, encoder_input_padded, decoder_input_padded, decoder_input_lengths,
                    decoder_targets, batch_size):
    # 1.1.2.4
    # Subdivision in train, validation and test set
    train_size = int((total_num_sentences / 100) * partitions[0])
    validation_size = int((total_num_sentences / 100) * partitions[1])
    test_size = int((total_num_sentences / 100) * partitions[2])

    # Can I improve this splitting with tf.dataset.shuffle, tf.take and tf.skip ?
    # In this manual way I am sure that the train and validation sets do not contain elements from the test set
    # Moreover, with ramdom.permutation I am sure that we cannot introduce a bias into the system
    # See https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets for more information

    # 1.1.2.5
    # generate three datasets. They contain the encoder input, decoder input, decoder input length and decodeer targets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (encoder_input_padded[0:train_size],
         decoder_input_padded[0:train_size],
         decoder_input_lengths[0:train_size],
         decoder_targets[0:train_size])
    ).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # this last line allows to have only batch of the same size.
    # In new TF version I should use .batch(batch_size, drop_remainder=True) but in the cluster there is an older v

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (encoder_input_padded[train_size:train_size + validation_size],
         decoder_input_padded[train_size:train_size + validation_size],
         decoder_input_lengths[train_size:train_size + validation_size],
         decoder_targets[train_size:train_size + validation_size])
    ).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (encoder_input_padded[train_size + validation_size:],
         decoder_input_padded[train_size + validation_size:],
         decoder_input_lengths[train_size + validation_size:],
         decoder_targets[train_size + validation_size:])
    ).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    # 1.1.2.3
    # shuffle and make repeat the dataset. The test set does not repeat so that it is execute only once
    train_dataset = train_dataset.shuffle(buffer_size=train_size).repeat()
    validation_dataset = validation_dataset.shuffle(buffer_size=validation_size).repeat()
    test_dataset = test_dataset.shuffle(buffer_size=test_size)

    return train_dataset, validation_dataset, test_dataset


def create_encoder(voc_size, input_data, embedding_size, hidden_size, keep_probability):
    # 1.2.1.2
    # create embedded input for encoder. This allows to not have an input vector of size len(input_voc)
    embedded_input = tf.contrib.layers.embed_sequence(input_data,
                                                      vocab_size=voc_size,
                                                      embed_dim=embedding_size)

    # 1.2.1.3
    # Dropout wrapper for dropout see https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f
    rnn_encoder_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units=hidden_size), keep_probability)

    # create rnn and conserve outputs for decoder
    encoder_output, encoder_final_state = tf.nn.dynamic_rnn(rnn_encoder_cell,
                                                            embedded_input,
                                                            dtype=tf.float32)
    return encoder_output, encoder_final_state


def compute_loss(decoder_output, decoder_input_max_len, decoder_output_lengths, decoder_input_lengths, decoder_targets):
    # 1.2.2.8 and # 1.2.3.3
    # get logits from cell output
    logits = tf.identity(decoder_output.rnn_output)
    # pad logits to same length (max output language length)
    logits_pad = tf.pad(
        logits,
        [[0, 0],
         # do -1?
         [0, tf.maximum(decoder_input_max_len - tf.reduce_max(decoder_output_lengths), 0)],
         # this pad all the sequences to the max length. It add x zeros to each logits
         # decoder_input_max_len - tf.reduce_max(decoder_output_lengths) this tells how many zero to add
         [0, 0]],
        mode='CONSTANT')

    # 1.2.2.9
    # get loss from spare softmax cross entropy
    # target do not need to be one_hot encoded (they need to be only normal integers)
    # /!\ the decoder expect target without the SOS value /!\
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_targets, logits=logits_pad)

    # 1.2.2.10
    # create mask to not count word outside target lengths. We do not care about extra word.
    # subtract 1 because the target are 1 smaller than decoder input
    mask = tf.sequence_mask(decoder_input_lengths, maxlen=decoder_input_max_len, dtype=tf.float32)
    loss = tf.boolean_mask(loss, mask)
    # Reduce the loss to single value
    loss = tf.reduce_mean(loss)

    # 1.2.3.4
    # get prediction with argmax on the second dim ([batch_size,sentence_length,output_words]
    prediction = tf.argmax(logits_pad, 2, output_type=tf.int32, name="prediction")
    # compute accuracy using mask to count only relevant words
    accuracy = tf.reduce_mean(tf.cast(tf.boolean_mask(tf.equal(decoder_targets, prediction), mask), tf.float32))

    return loss, prediction, accuracy


def create_decoder_layer(encoder_output, encoder_final_state, decoder_input_voc, decoder_input, decoder_input_lengths,
                         decoder_input_max_len, decoder_targets, embedding_size, batch_size, hidden_size,
                         keep_probability):
    # decoding language size
    voc_size = len(decoder_input_voc)

    # 1.2.2.1
    # embedding variable with uniform data values
    embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size]))
    # embedded decoder input. As said this allow to have smaller input vector and it performs better
    # https://www.tensorflow.org/guide/embedding
    embedded_decoder_input = tf.nn.embedding_lookup(embeddings, decoder_input)

    # 1.2.2.2
    # Create LSTM cell and attention mechanism to be able to look at the decoder input values
    rnn_decoder_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size)
    attention_mec = tf.contrib.seq2seq.BahdanauAttention(num_units=hidden_size, memory=encoder_output)
    decorated_rnn_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=rnn_decoder_cell,
                                                                     attention_mechanism=attention_mec,
                                                                     alignment_history=True)

    # 1.2.2.3
    # Create initial state for decoder from the output state of the encoder
    initial_decoder_state = decorated_rnn_decoder_cell.zero_state(batch_size, dtype=tf.float32).clone(
        cell_state=encoder_final_state)

    # 1.2.2.5
    # output dense layer
    output_layer = tf.layers.Dense(voc_size)

    # create decoder for training
    decoder_output, decoder_output_lengths = create_decoder(embedded_decoder_input,
                                                            decorated_rnn_decoder_cell,
                                                            initial_decoder_state,
                                                            output_layer,
                                                            decoder_input_max_len,
                                                            decoder_input_lengths,
                                                            keep_probability)

    # create inference decoder for prediction
    inference_decoder_output, inference_dec_states, inference_decoder_output_lengths = create_inference_decoder(
        embeddings,
        decoder_input_voc,
        decorated_rnn_decoder_cell,
        initial_decoder_state,
        output_layer,
        decoder_input_max_len,
        keep_probability,
        batch_size)
    # get loss and accuracy for training
    train_decoder_loss, _, train_decoder_accuracy = compute_loss(decoder_output,
                                                                 decoder_input_max_len,
                                                                 decoder_output_lengths,
                                                                 decoder_input_lengths,
                                                                 decoder_targets)

    # get predictions and inference accuracy
    _, prediction, inference_decoder_accuracy = compute_loss(inference_decoder_output,
                                                             decoder_input_max_len,
                                                             inference_decoder_output_lengths,
                                                             decoder_input_lengths,
                                                             decoder_targets)

    return train_decoder_loss, prediction, train_decoder_accuracy, inference_decoder_accuracy, inference_dec_states


def create_decoder(embedded_decoder_input, cell, initial_state, output_layer, decoder_input_max_len,
                   decoder_input_lengths, keep_probability):
    # Dropout wrapper see https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_probability)

    # 1.2.2.4
    # Training helper which gives at each time step the input (teach forcing). This allow to have better training
    # If we use a greedy helper to pick the output of the previous step it is really bad at the beginning and therefore
    # the network would not be able to learn
    helper = tf.contrib.seq2seq.TrainingHelper(embedded_decoder_input, decoder_input_lengths)

    # 1.2.2.6 and 1.2.3.2
    # create basic decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer)

    # 1.2.2.7
    # get outputs (keeping the lengths for padding)
    decoder_output, _, decoder_output_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                                                  maximum_iterations=decoder_input_max_len)

    return decoder_output, decoder_output_lengths


def create_inference_decoder(embeddings, decoder_input_voc, cell, initial_state, output_layer, decoder_input_max_len,
                             keep_probability, batch_size):
    # Dropout wrapper see https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_probability)

    # 1.2.3.1
    # use greedy helper for inference. It picks the highest probability from the output of the previous step
    # Previous step is fed into next one
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, tf.fill([batch_size], decoder_input_voc.index(SOS)),
                                                      decoder_input_voc.index(EOS))
    # 1.2.2.6 and 1.2.3.2
    # create basic decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer)

    # 1.2.2.7
    # get outputs (keeping the lengths for padding)
    inference_decoder_output, inference_dec_states, inference_decoder_output_lengths = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        impute_finished=True,
        maximum_iterations=decoder_input_max_len)

    return inference_decoder_output, inference_dec_states, inference_decoder_output_lengths


def train_model(eng_voc, spa_voc, targets_max_len, input_max_len, decoder_input_lengths, decoder_targets,
                decoder_input_padded,
                encoder_input_padded, number_of_sentences, encoder_max_len):
    # hyper parameters
    batch_size = 64
    embedding_size = 256
    hidden_size = 1024
    total_num_batches = 6100
    learning_rate = 0.001

    # create data sets
    train_dataset, validation_dataset, test_dataset = create_data_set(
        [65, 15, 20],  # this is the distribution for train, validation and test set
        number_of_sentences,
        encoder_input_padded,
        decoder_input_padded,
        decoder_input_lengths,
        decoder_targets,
        batch_size)

    # 1.2.1.1
    # Graph placeholders
    encoder_input = tf.placeholder(dtype=tf.int32, shape=(batch_size, input_max_len), name="encoder_input")
    decoder_input = tf.placeholder(dtype=tf.int32, shape=(batch_size, targets_max_len), name="decoder_input")
    decoder_targets = tf.placeholder(dtype=tf.int32, shape=(batch_size, targets_max_len), name="decoder_targets")
    decoder_input_lengths = tf.placeholder(dtype=tf.int32, shape=batch_size, name="decoder_input_lengths")
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")

    # create encoder
    encoder_output, encoder_final_state = create_encoder(len(spa_voc),
                                                         encoder_input,
                                                         embedding_size,
                                                         hidden_size,
                                                         keep_probability)

    # create decoder layer
    train_decoder_loss, prediction, train_decoder_accuracy, inference_decoder_accuracy, inference_dec_states = create_decoder_layer(
        encoder_output,
        encoder_final_state,
        eng_voc,
        decoder_input,
        decoder_input_lengths,
        targets_max_len,
        decoder_targets,
        embedding_size,
        batch_size,
        hidden_size,
        keep_probability)

    # 1.3.1
    # optimiter and train for network
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(train_decoder_loss)

    # 1.3.2
    print("Graph initialization")

    # initialize graph
    session.run(tf.global_variables_initializer())

    # generate iterators from datasets
    train_iterator = train_dataset.make_one_shot_iterator()
    train_batch = train_iterator.get_next()

    validation_iterator = validation_dataset.make_one_shot_iterator()
    validation_batch = validation_iterator.get_next()

    print("Starting train")

    train_loss = []
    train_accuracy = []
    validation_accuracy = []
    # Loop over batches
    for batch in range(total_num_batches):

        # get batch values from training dataset
        encoder_input_val, decoder_input_val, decoder_input_len_val, decoder_targets_val = session.run(train_batch)
        # train network and get cost and accuracy
        loss, accuracy, _ = session.run([train_decoder_loss, train_decoder_accuracy, train],
                                        {decoder_input: decoder_input_val,
                                         encoder_input: encoder_input_val,
                                         decoder_input_lengths: decoder_input_len_val,
                                         decoder_targets: decoder_targets_val,
                                         keep_probability: 1.0})
        # append loss and accuracy to make mean after 100 batches
        train_loss.append(loss)
        train_accuracy.append(accuracy)

        # get batch values from validation dataset
        encoder_input_val, decoder_input_val, decoder_input_len_val, decoder_targets_val = session.run(validation_batch)
        # get validation accuracy
        a = session.run([inference_decoder_accuracy],
                        {decoder_input: decoder_input_val,
                         encoder_input: encoder_input_val,
                         decoder_input_lengths: decoder_input_len_val,
                         decoder_targets: decoder_targets_val,
                         keep_probability: 1.0})
        # append accuracy to make mean after 100 batches
        validation_accuracy.append(a)

        # print information after 100 batches
        if batch % 100 == 0:
            print(
                "Batch {0} train loss: {1:.3f}, train accuracy: {2:.3f}, validation accuracy: {3:.3f} ".format(batch,
                                                                                                               np.mean(
                                                                                                                   train_loss),
                                                                                                               np.mean(
                                                                                                                   train_accuracy),
                                                                                                               np.mean(
                                                                                                                   validation_accuracy)))
            # clean lists
            train_loss = []
            train_accuracy = []
            validation_accuracy = []

        if batch % 1000 == 0:
            translate("¿todavía están en casa?", prediction, keep_probability, encoder_input, encoder_max_len, spa_voc,
                      eng_voc, batch_size, inference_dec_states, batch)

    # Do same thing for test set once the training is done
    test_iterator = test_dataset.make_one_shot_iterator()
    test_batch = test_iterator.get_next()

    test_accuracy = []
    # loop until the iterator does not have anymore batches
    while True:
        try:
            encoder_input_val, decoder_input_val, decoder_input_len_val, decoder_targets_val = session.run(test_batch)
            a = session.run([inference_decoder_accuracy],
                            {decoder_input: decoder_input_val,
                             encoder_input: encoder_input_val,
                             decoder_input_lengths: decoder_input_len_val,
                             decoder_targets: decoder_targets_val,
                             keep_probability: 1.0})
            test_accuracy.append(a)
        except tf.errors.OutOfRangeError:
            break

    print("Test accuracy: {0:.3f}".format(np.mean(test_accuracy)))

    return prediction, keep_probability, encoder_input, batch_size, inference_dec_states


# 1.1.2.1
# compute lengths of sequences
def sentences_sizes(sequences):
    sizes = []
    for s in sequences:
        sizes.append(len(s))

    return np.asarray(sizes, dtype=np.int32)


# 1.1.1.5
# see assignment description
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# preprocess sequence to format text, split and add SOS and EOS
# SOS is omitted for targets
def preprocess_sequences(sentences, target=False):
    # 1.1.1.4
    # lowercase
    sentences = [s.lower() for s in sentences]

    # 1.1.1.5
    # to ascii see assignemnt description
    sentences = [unicode_to_ascii(s) for s in sentences]

    # 1.1.1.6
    # Add spaces between special characters
    pat = re.compile(r"([?.!,¿¡])")
    # remove extra white spaces
    sentences = [pat.sub(" \\1 ", s) for s in sentences]
    # remove new lines
    sentences = [s.replace('\n', ' ') for s in sentences]

    # 1.1.1.7
    # add SOS and EOS (SOS is omitted for targets)
    if target:
        sentences = [s + " " + EOS for s in sentences]
    else:
        sentences = [SOS + " " + s + " " + EOS for s in sentences]

    # 1.1.1.6
    # Removing white spaces here because the line above add multiple of them :S
    sentences = [re.sub(' +', ' ', s) for s in sentences]

    # 1.1.1.8
    # split sentences by space
    sentences = [s.split(' ') for s in sentences]

    return sentences


# 1.1.1.10
# create dictionary containing all the words (PAD is at element 0)
def create_vocabulary(sequences):
    vocabulary = [PAD]
    for s in sequences:
        for word in s:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


# 1.1.1.10
# transform word to integer using the given dictionary (it uses the indexes)
def word2int(vocabulary, sequences):
    numbers = []
    for s in sequences:
        sentence = []
        for word in s:
            sentence.append(vocabulary.index(word))
        numbers.append(sentence)
    return numbers


# 1.1.1.10
# transform int to word using the given dictionary (it take the value at index int)
def int2word(vocabulary, numbers):
    sequences = []
    for s in numbers:
        sentence = []
        for word in s:
            sentence.append(vocabulary[word])
        sequences.append(sentence)
    return sequences


# 1.1.1.10
# get statistical value from dictionary and print them
def perform_statistics(eng_voc, spa_voc, eng_seqs, spa_seqs, outout=False):
    max_len_eng = np.max([len(s) for s in eng_seqs])
    max_len_spa = np.max([len(s) for s in spa_seqs])
    eng_voc_size = len(eng_voc)
    spa_voc_size = len(spa_voc)

    if outout:
        print("There are a total of {0} words in the english vocabulary".format(eng_voc_size))
        print("There are a total of {0} words in the spanish vocabulary".format(spa_voc_size))
        print("The max size of a english sentence is {0}".format(max_len_eng))
        print("The max size of a spanish sentence is {0}".format(max_len_spa))

    return max_len_eng, max_len_spa


# 1.3.5
# function to save an image which show the attention mechanism (see assignment description)
def store_attention_plot(attn_map, input_tokens, output_tokens, step_id):
    input_len = len(input_tokens)
    output_len = len(output_tokens)
    attn_map = attn_map[:, :output_len]
    attn_map = attn_map.T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(attn_map, interpolation="nearest", cmap="Blues")

    ax.set_yticks(range(input_len))
    ax.set_yticklabels(input_tokens)

    ax.set_xticks(range(output_len))
    ax.set_xticklabels(output_tokens)

    ax.set_ylabel("input sequence")
    ax.set_xlabel("output sequence")

    ax.grid()

    plt.savefig("alignment-{}.png".format(step_id), bbox_inches="tight")


def translate(input_sentence, prediction, keep_probability, encoder_input, max_len_spa, spa_voc, eng_voc, batch_size,
              inference_dec_states, step=0):
    # preprocess input sequence
    sentence = preprocess_sequences([input_sentence])

    # transform sequence to int values
    int_sentence = word2int(spa_voc, sentence)

    # pad sequence to eng voc max length
    int_sentence = tf.keras.preprocessing.sequence.pad_sequences(int_sentence, maxlen=max_len_spa, padding='post')

    # create input vector as in the training of the network (not change in sizes)
    encoder_input_val = np.zeros(shape=(batch_size, max_len_spa), dtype=np.int32)

    # insert into input vector the input sentence
    encoder_input_val[0] = int_sentence[0]
    # get prediction
    final_pred, alignments = session.run([prediction, inference_dec_states.alignment_history.stack()],
                                         {encoder_input: encoder_input_val,
                                          keep_probability: 1.0})

    output = int2word(eng_voc, final_pred)[0]
    # remove pad simbols
    output = re.sub(' +', ' ', ' '.join(v if v != PAD else '' for v in output)).strip()

    # print translation
    print("At step {0} the system translate '{1}' to '{2}'".format(step, input_sentence, output))

    store_attention_plot(alignments[:len(sentence[0]), 0, :len(output.split(" "))], sentence[0], output.split(" "), step)


def train():
    # number of sentences for the training
    number_of_sentences = 30000
    # generate data for training
    eng_voc, spa_voc, max_len_eng, max_len_spa, eng_sizes, eng_padded, spa_padded, target = read_and_prepare_data(
        number_of_sentences)
    # train model
    prediction, keep_probability, encoder_input, batch_size, inference_dec_states = train_model(eng_voc, spa_voc,
                                                                                                max_len_eng,
                                                                                                max_len_spa,
                                                                                                eng_sizes,
                                                                                                target, eng_padded,
                                                                                                spa_padded,
                                                                                                number_of_sentences,
                                                                                                max_len_spa)

    return prediction, keep_probability, encoder_input, max_len_spa, spa_voc, eng_voc, batch_size, inference_dec_states


# should be better to save and load model after training
prediction, keep_probability, encoder_input, max_len_spa, spa_voc, eng_voc, batch_size, inference_dec_states = train()
translate("¿todavía están en casa?", prediction, keep_probability, encoder_input, max_len_spa, spa_voc, eng_voc,
          batch_size, inference_dec_states)
