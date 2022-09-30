from collections import Counter, defaultdict
import numpy as np
import torch
import os
import json
import gensim
import logging
import pickle
from torch.utils.data import Dataset
import scipy.sparse as sp
import gc
from tree_Operate import *

event_roles = {
    'ef': {'other':0,'B_EquityHolder': 1, 'I_EquityHolder': 2, 'B_FrozeShares': 3, 'I_FrozeShares': 4, 'B_LegalInstitution': 5,
           'I_LegalInstitution': 6, 'B_TotalHoldingShares': 7, 'I_TotalHoldingShares': 8, 'B_TotalHoldingRatio': 9,
           'I_TotalHoldingRatio': 10, 'B_StartDate': 11, 'I_StartDate': 12, 'B_EndDate': 13, 'I_EndDate': 14,
           'B_UnfrozeDate': 15, 'I_UnfrozeDate': 16},
    'er': {'other':0,'B_CompanyName': 1, 'I_CompanyName': 2, 'B_HighestTradingPrice': 3, 'I_HighestTradingPrice': 4,
           'B_LowestTradingPrice': 5, 'I_LowestTradingPrice': 6, 'B_RepurchasedShares': 7, 'I_RepurchasedShares': 8,
           'B_ClosingDate': 9, 'I_ClosingDate': 10, 'B_RepurchaseAmount': 11, 'I_RepurchaseAmount': 12},
    'eu': {'other':0,'B_EquityHolder': 1, 'I_EquityHolder': 2, 'B_TradedShares': 3, 'I_TradedShares': 4, 'B_StartDate': 5,
           'I_StartDate': 6, 'B_EndDate': 7, 'I_EndDate': 8, 'B_LaterHoldingShares': 9, 'I_LaterHoldingShares': 10,
           'B_AveragePrice': 11, 'I_AveragePrice': 12},
    'eo': {'other':0,'B_EquityHolder': 1, 'I_EquityHolder': 2, 'B_TradedShares': 3, 'I_TradedShares': 4, 'B_StartDate': 5,
           'I_StartDate': 6, 'B_EndDate': 7, 'I_EndDate': 8, 'B_LaterHoldingShares': 9, 'I_LaterHoldingShares': 10,
           'B_AveragePrice': 11, 'I_AveragePrice': 12},
    'ep': {'other':0,'B_Pledger': 1, 'I_Pledger': 2, 'B_PledgedShares': 3, 'I_PledgedShares': 4, 'B_Pledgee': 5, 'I_Pledgee': 6,
           'B_TotalHoldingShares': 7, 'I_TotalHoldingShares': 8, 'B_TotalHoldingRatio': 9, 'I_TotalHoldingRatio': 10,
           'B_TotalPledgedShares': 11, 'I_TotalPledgedShares': 12, 'B_StartDate': 13, 'I_StartDate': 14,
           'B_EndDate': 15, 'I_EndDate': 16, 'B_ReleasedDate': 17, 'I_ReleasedDate': 18}}

logger = logging.getLogger(__name__)

def load_datasets_and_vocabs(args):
    train_example_file = os.path.join(args.cache_dir, 'train_example.pkl')
    test_example_file = os.path.join(args.cache_dir, 'test_example.pkl')
    train_weight_file = os.path.join(args.cache_dir, 'train_weight_catch.txt')
    test_weight_file = os.path.join(args.cache_dir, 'test_weight_catch.txt')

    if os.path.exists(train_example_file) and os.path.exists(test_example_file):
        logger.info('Loading train_example from %s', train_example_file)
        with open(train_example_file, 'rb') as f:
            train_examples = pickle.load(f)

        logger.info('Loading test_example from %s', test_example_file)
        with open(test_example_file, 'rb') as f:
            test_examples = pickle.load(f)

        with open(train_weight_file, 'rb') as f:
            train_labels_weight = torch.Tensor(json.load(f))
        with open(test_weight_file, 'rb') as f:
            test_labels_weight = torch.Tensor(json.load(f))
    else:
        train_tree_file = os.path.join(args.dataset_path,'train.pkl')
        test_tree_file = os.path.join(args.dataset_path,'test.pkl')
        logger.info('Loading train trees')

        with open(train_tree_file, 'rb') as f:
            train_trees = pickle.load(f)

        # get examples of data
        train_examples,train_labels_weight = create_example(train_trees,args.train_event_num)
        logger.info('Creating train examples')
        with open(train_example_file, 'wb') as f:
            pickle.dump(train_examples, f, -1)

        logger.info('Loading test trees')
        with open(test_tree_file, 'rb') as f:
            test_trees = pickle.load(f)
        test_examples,test_labels_weight = create_example(test_trees,args.test_event_num)

        logger.info('Creating test examples')
        with open(test_example_file,'wb') as f:
            pickle.dump(test_examples,f,-1)

        with open(train_weight_file,'w') as wf:
            json.dump(train_labels_weight,wf)
        logger.info('Creating test_weight_cache')
        with open(test_weight_file,'w') as wf:
            json.dump(test_labels_weight,wf)

    logger.info('Train set size: %s', len(train_examples))
    logger.info('Test set size: %s,', len(test_examples))

    # Build word vocabulary(dep_tag, part of speech) and save pickles.
    word_vecs,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab = load_and_cache_vocabs(train_examples+test_examples, args)

    if args.embedding_type == 'word2vec':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32)).squeeze(1)
        args.word2vec_embedding = embedding

    train_dataset = ED_Dataset(train_examples,args,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab)
    test_dataset = ED_Dataset(test_examples,args,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab)

    return train_dataset,train_labels_weight,test_dataset,test_labels_weight,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab

def create_example(docs,event_num):
    examples = []
    ef_labels_ids = []
    er_labels_ids = []
    eu_labels_ids = []
    eo_labels_ids = []
    ep_labels_ids = []
    for i,doc in enumerate(docs):
        example = {'t_ids':[],'tokens':[],'pos':[],'deps':[],'sen_pos':[],'word_pos':[],'parents':[],'ppos':[],'pdeps':[],'e_ids':[],'ef_labels':[],'er_labels':[],
                   'eu_labels':[],'eo_labels':[],'ep_labels':[]}

        nodes = doc.dp_tree.all_nodes()
        nodes.sort(key=doc.node_sort)
        for node in nodes:
            if node.identifier == DROOT:
                continue
            for i in range(event_num):
                example['tokens'].append(node.tag)
                example['pos'].append(node.data.pos)
                example['deps'].append(node.data.dep)
                example['sen_pos'].append(node.data.sen_pos)
                example['word_pos'].append(node.data.token_id)
                example['ef_labels'].append(int(node.data.token_labels['ef'][i]))
                example['er_labels'].append(int(node.data.token_labels['er'][i]))
                example['eu_labels'].append(int(node.data.token_labels['eu'][i]))
                example['eo_labels'].append(int(node.data.token_labels['eo'][i]))
                example['ep_labels'].append(int(node.data.token_labels['ep'][i]))
                pnode = doc.dp_tree.parent(node.identifier)
                example['parents'].append(pnode.tag)
                example['ppos'].append(pnode.data.pos)
                example['pdeps'].append(pnode.data.dep)
                example['e_ids'].append(i)

        examples.append(example)
        ef_labels_ids += example['ef_labels']
        er_labels_ids += example['er_labels']
        eu_labels_ids += example['eu_labels']
        eo_labels_ids += example['eo_labels']
        ep_labels_ids += example['ep_labels']

    ef_label_weight = get_labels_weight(ef_labels_ids, event_roles['ef'])
    er_label_weight = get_labels_weight(er_labels_ids, event_roles['er'])
    eu_label_weight = get_labels_weight(eu_labels_ids, event_roles['eu'])
    eo_label_weight = get_labels_weight(eo_labels_ids, event_roles['eo'])
    ep_label_weight = get_labels_weight(ep_labels_ids, event_roles['ep'])
    labels_weight = ef_label_weight + er_label_weight + eu_label_weight + eo_label_weight + ep_label_weight
    return examples,labels_weight

def get_labels_weight(label_ids,labels_lookup):
    nums_labels = Counter(label_ids)
    nums_labels = [(l,k) for k, l in sorted([(j, i) for i, j in nums_labels.items()], reverse=True)]
    size = len(nums_labels)
    if size % 2 == 0:
        median = (nums_labels[size // 2][1] + nums_labels[size//2-1][1])/2
    else:
        median = nums_labels[(size - 1) // 2][1]

    weight_list = []
    for value_id in labels_lookup.values():
        if value_id not in label_ids:
            weight_list.append(0)
        else:
            for label in nums_labels:
                if label[0] == value_id:
                    weight_list.append(median/label[1])
                    break
    return weight_list

def remove_repetion(llist):
    new_list = []
    for li in llist:
        if li not in new_list:
            new_list.append(li)
    return new_list

def build_adj(sour_edges,t_ids):
    ids = np.array(t_ids, dtype=np.int32)
    matrix_shape = np.array(t_ids).shape[0]

    idx_map = {j: i for i, j in enumerate(ids)}
    edges = []
    for i,edge in enumerate(sour_edges):
        edges.append([idx_map[edge[0]], idx_map[edge[1]]])
    edges = np.array(edges, dtype=np.int32).reshape(np.array(sour_edges).shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(matrix_shape, matrix_shape),dtype=np.float32)

    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj

def load_and_cache_vocabs(examples,args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    embedding_cache_path = os.path.join(args.cache_dir, 'embedding')
    if not os.path.exists(embedding_cache_path):
        os.makedirs(embedding_cache_path)

    # Build or load word vocab and word2vec embeddings.
    cached_word_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_word_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_vocab_file):
        logger.info('Loading word vocab from %s', cached_word_vocab_file)
        with open(cached_word_vocab_file, 'rb') as f:
            word_vocab = pickle.load(f)
    else:
        logger.info('Creating word vocab from dataset %s',args.dataset_name)
        word_vocab = build_text_vocab(examples)
        logger.info('Word vocab size: %s', word_vocab['len'])
        logging.info('Saving word vocab to %s', cached_word_vocab_file)
        with open(cached_word_vocab_file, 'wb') as f:
            pickle.dump(word_vocab, f, -1)

    cached_word_vecs_file = os.path.join(embedding_cache_path, 'cached_{}_word_vecs.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_vecs_file):
        logger.info('Loading word vecs from %s', cached_word_vecs_file)
        with open(cached_word_vecs_file, 'rb') as f:
            word_vecs = pickle.load(f)
    else:
        logger.info('Creating word vecs from %s', args.embedding_dir)
        word_vecs = load_word2vec_embedding(word_vocab['itos'], args, 0.25)
        logger.info('Saving word vecs to %s', cached_word_vecs_file)
        with open(cached_word_vecs_file, 'wb') as f:
            pickle.dump(word_vecs, f, -1)

    # Build vocab of dep tags.
    cached_dep_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dep tags from %s', cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dep tags.')
        dep_tag_vocab = build_dep_tag_vocab(examples, min_freq=0)
        logger.info('Saving dep tags  vocab, size: %s, to file %s', dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of pos tags from %s',cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of pos tags.')
        pos_tag_vocab = build_pos_tag_vocab(examples, min_freq=0)
        logger.info('Saving pos tags  vocab, size: %s, to file %s',pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    # Build vocab of sentence position tags.
    cached_sen_pos_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_sen_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_sen_pos_tag_vocab_file):
        logger.info('Loading vocab of sentence pos tags from %s', cached_sen_pos_tag_vocab_file)
        with open(cached_sen_pos_tag_vocab_file, 'rb') as f:
            sen_pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of sentence pos tags.')
        sen_pos_tag_vocab = build_sen_pos_tag_vocab(examples, min_freq=0)
        logger.info('Saving sentence pos tags  vocab, size: %s, to file %s', sen_pos_tag_vocab['len'], cached_sen_pos_tag_vocab_file)
        with open(cached_sen_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(sen_pos_tag_vocab, f, -1)

    # Build vocab of word position tags.
    cached_word_pos_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_word_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_pos_tag_vocab_file):
        logger.info('Loading vocab of word pos tags from %s', cached_word_pos_tag_vocab_file)
        with open(cached_word_pos_tag_vocab_file, 'rb') as f:
            word_pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of word pos tags.')
        word_pos_tag_vocab = build_word_pos_tag_vocab(examples, min_freq=0)
        logger.info('Saving word pos tags  vocab, size: %s, to file %s', word_pos_tag_vocab['len'],
                    cached_word_pos_tag_vocab_file)
        with open(cached_word_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(word_pos_tag_vocab, f, -1)

    # Build vocab of event ids tags.
    cached_event_id_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_event_id_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_event_id_tag_vocab_file):
        logger.info('Loading vocab of event id tags from %s', cached_event_id_tag_vocab_file)
        with open(cached_event_id_tag_vocab_file, 'rb') as f:
            event_id_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of event id tags.')
        event_id_tag_vocab = build_event_id_tag_vocab(examples, min_freq=0)
        logger.info('Saving event id tags  vocab, size: %s, to file %s', event_id_tag_vocab['len'],
                    cached_event_id_tag_vocab_file)
        with open(cached_event_id_tag_vocab_file, 'wb') as f:
            pickle.dump(event_id_tag_vocab, f, -1)

    return word_vecs,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab

def load_word2vec_embedding(words,args,uniform_scale):

    path = os.path.join(args.embedding_dir,'baike_26g_news_13g_novel_229g.model')
    w2v_model = gensim.models.Word2Vec.load(path)

    w2v_vocabs = [word for word, Vocab in w2v_model.wv.vocab.items()]

    word_vectors = []
    for word in words:
        if word in w2v_vocabs:
            word_vectors.append(w2v_model.wv[word])
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, w2v_model.vector_size))
    return word_vectors


def _default_unk_index():
    return 1

def build_text_vocab(examples, vocab_size=1000000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['tokens']+example['parents'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_dep_tag_vocab(examples, vocab_size=1000, min_freq=0):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for example in examples:
        counter.update(example['deps']+example['pdeps'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_pos_tag_vocab(examples, vocab_size=1000, min_freq=0):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for example in examples:
        counter.update(example['pos']+example['ppos'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_sen_pos_tag_vocab(examples, vocab_size=1000, min_freq=0):
    """
    sentence id tags vocab.
    """
    counter = Counter()
    for example in examples:
        counter.update(example['sen_pos'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_word_pos_tag_vocab(examples, vocab_size=10000, min_freq=0):
    """
    sentence id tags vocab.
    """
    counter = Counter()
    for example in examples:
        counter.update(example['word_pos'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_event_id_tag_vocab(examples, vocab_size=10000, min_freq=0):
    """
    sentence id tags vocab.
    """
    counter = Counter()
    for example in examples:
        counter.update(example['e_ids'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

class ED_Dataset(Dataset):
    def __init__(self, examples,args,word_vocab,pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab):
        self.examples = examples
        self.args = args
        self.word_vocab = word_vocab
        self.pos_tag_vocab = pos_tag_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.sen_pos_tag_vocab = sen_pos_tag_vocab
        self.word_pos_tag_vocab = word_pos_tag_vocab
        self.event_id_tag_vocab = event_id_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        items = e['token_ids'],e['pos_ids'],e['dep_ids'],e['sen_pos_ids'],e['word_pos_ids'],e['parent_ids'],e['ppos_ids'],e['pdep_ids'],e['event_ids'],\
                e['ef_labels'], e['er_labels'],e['eu_labels'],e['eo_labels'],e['ep_labels']

        items_tensor = tuple(torch.tensor(t) for t in items)
        return items_tensor

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''

        for i in range(len(self.examples)):
            self.examples[i]['token_ids'] = [self.word_vocab['stoi'][w] for w in self.examples[i]['tokens']]
            self.examples[i]['parent_ids'] = [self.word_vocab['stoi'][w] for w in self.examples[i]['parents']]
            self.examples[i]['pos_ids'] = [self.pos_tag_vocab['stoi'][p] for p in self.examples[i]['pos']]
            self.examples[i]['ppos_ids'] = [self.pos_tag_vocab['stoi'][p] for p in self.examples[i]['ppos']]
            self.examples[i]['dep_ids'] = [self.dep_tag_vocab['stoi'][d] for d in self.examples[i]['deps']]
            self.examples[i]['pdep_ids'] = [self.dep_tag_vocab['stoi'][d] for d in self.examples[i]['pdeps']]
            self.examples[i]['sen_pos_ids'] = [self.sen_pos_tag_vocab['stoi'][sen_pos] for sen_pos in self.examples[i]['sen_pos']]
            self.examples[i]['word_pos_ids'] = [self.word_pos_tag_vocab['stoi'][word_pos] for word_pos in self.examples[i]['word_pos']]
            self.examples[i]['event_ids'] = [self.event_id_tag_vocab['stoi'][e] for e in
                                                 self.examples[i]['e_ids']]


def my_collate(batch):
    '''
    Pad event in a batch.
    Sort the events based on length.
    Turn all into tensors.
    '''
    # from Dataset.__getitem__()
    token_ids,pos_ids,dep_ids,sen_pos_ids,word_pos_ids,parent_ids,ppos_ids,pdep_ids,event_ids,\
    ef_labels,er_labels,eu_labels,eo_labels,ep_labels  = zip(
        *batch)  # from Dataset.__getitem__()

    token_ids = token_ids[0]
    pos_ids = pos_ids[0]
    dep_ids = dep_ids[0]
    sen_pos_ids = sen_pos_ids[0]
    word_pos_ids = word_pos_ids[0]
    parent_ids = parent_ids[0]
    ppos_ids = ppos_ids[0]
    pdep_ids = pdep_ids[0]
    event_ids = event_ids[0]
    ef_labels = ef_labels[0]
    er_labels = er_labels[0]
    eu_labels = eu_labels[0]
    eo_labels = eo_labels[0]
    ep_labels = ep_labels[0]

    return token_ids,pos_ids,dep_ids,sen_pos_ids,word_pos_ids,parent_ids,ppos_ids,pdep_ids,event_ids,\
           ef_labels,er_labels,eu_labels,eo_labels,ep_labels

