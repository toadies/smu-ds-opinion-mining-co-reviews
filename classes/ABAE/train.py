import numpy as np
from time import time
from tqdm import tqdm
import keras.backend as K


def train(model, sen_gen, neg_gen, epochs, batch_size, batches_per_epoch, result_path_name, vocab_inv):
    min_loss = float('inf')
    t0 = time()
    final_model = model
    for ii in range(epochs):
        loss, max_margin_loss = 0., 0.

        # for b in tqdm(range(batches_per_epoch)):
        for b in range(batches_per_epoch):
            sen_input = next(sen_gen)
            neg_input = next(neg_gen)

            batch_loss, batch_max_margin_loss = model.train_on_batch([sen_input, neg_input],
                                                                    np.ones((batch_size, 1)))
            loss += batch_loss / batches_per_epoch
            max_margin_loss += batch_max_margin_loss / batches_per_epoch

        

        if loss < min_loss:
            min_loss = loss
            word_emb = K.get_value(model.get_layer('word_emb').embeddings)
            aspect_emb = K.get_value(model.get_layer('aspect_emb').W)
            word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)
            aspect_emb = aspect_emb / np.linalg.norm(aspect_emb, axis=-1, keepdims=True)

            aspect = {}
            # final_model = model
            
            for ind in range(len(aspect_emb)):
                desc = aspect_emb[ind]
                sims = word_emb.dot(desc.T)
                ordered_words = np.argsort(sims)[::-1]
                desc_list = [vocab_inv[w] for w in ordered_words[:100]]
                # print('Aspect %d:' % ind)
                # print(desc_list)
                aspect['Aspect %d' % ind] = desc_list
        
    tr_time = time() - t0
    print('Epoch %d, train: %is' % (ii, tr_time))
    print('Total loss: %.4f, max_margin_loss: %.4f, ortho_reg: %.4f' % (loss, max_margin_loss, loss - max_margin_loss))
        
    return aspect, final_model
