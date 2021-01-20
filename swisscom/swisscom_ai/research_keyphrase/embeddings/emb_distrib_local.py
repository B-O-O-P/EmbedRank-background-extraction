# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
# Authors: Kamil Bennani-Smires, Yann Savary

import sys

sys.path.append("..")
from sentence_transformers import SentenceTransformer
from swisscom.swisscom_ai.research_keyphrase.embeddings.emb_distrib_interface import EmbeddingDistributor
import sent2vec


class EmbeddingDistributorLocal(EmbeddingDistributor):
    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec
    
    """

    def __init__(self, model_name, is_static=False):
        self.is_static = is_static
        if is_static:
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(model_name)
        else:
            self.model = SentenceTransformer(model_name)

    def get_tokenized_sents_embeddings(self, sents):
        """
        @see EmbeddingDistributor
        """
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        if self.is_static:
            return self.model.embed_sentences(sents)
        return self.model.encode(sentences=sents)
