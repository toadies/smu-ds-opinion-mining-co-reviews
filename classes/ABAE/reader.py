import codecs
import re
import operator
from collections import Counter


num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(token):
    return bool(num_regex.match(token))


def create_vocab(corpus, maxlen=0, word_freq=0):
    vocab = Counter()

    for line in corpus:
        words = line.split()
        
        for w in words:
            if is_number(w):
                pass
            else:
                vocab[w] += 1
    
    #Remove Words Repeating less then word_freq times
    final_list = [word for word in vocab if vocab[word] > word_freq]

    print("Unique Workds: {0}, Words after word_freq filter of {1}: {2}" \
          .format(len(vocab),word_freq, len(final_list)))
    
    vocab = { word:i for i, word in enumerate(final_list, 3)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    vocab["<num>"] = 2
    
    return vocab


def read_dataset(corpus, vocab, maxlen):
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    print("Corpus Size", len(corpus))
    x = 0
    for line in corpus:
        words = line.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            words = words[:maxlen]
        if not len(words):
            continue

        indices = []
        for word in words:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)
        x += 1

    print("Total Document Analyzed", x)
    print('<num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total)) # 
    return data_x, maxlen_x


def get_data(corpus, maxlen=0, word_freq=0):
    print(' Creating vocab ...')
    vocab = create_vocab(corpus, maxlen, word_freq)
    print(' Reading dataset ...')
    print('  train set')
    train_x, train_maxlen = read_dataset(corpus, vocab, maxlen)
    maxlen = train_maxlen
    return vocab, train_x, maxlen


if __name__ == "__main__":
    pass
