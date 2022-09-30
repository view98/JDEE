import logging
import json
from ltp import LTP
import pickle
import numpy as np
from tree_Operate import *
logger = logging.getLogger(__name__)

event_type_f2s = {'EquityFreeze': 'ef', 'EquityRepurchase': 'er', 'EquityUnderweight': 'eu', 'EquityOverweight': 'eo',
                  'EquityPledge': 'ep'}
event_roles = {
    'ef': {'B_EquityHolder': 0, 'I_EquityHolder': 1, 'B_FrozeShares': 2, 'I_FrozeShares': 3, 'B_LegalInstitution': 4,
           'I_LegalInstitution': 5, 'B_TotalHoldingShares': 6, 'I_TotalHoldingShares': 7, 'B_TotalHoldingRatio': 8,
           'I_TotalHoldingRatio': 9, 'B_StartDate': 10, 'I_StartDate': 11, 'B_EndDate': 12, 'I_EndDate': 13,
           'B_UnfrozeDate': 14, 'I_UnfrozeDate': 15},
    'er': {'B_CompanyName': 0, 'I_CompanyName': 1, 'B_HighestTradingPrice': 2, 'I_HighestTradingPrice': 3,
           'B_LowestTradingPrice': 4, 'I_LowestTradingPrice': 5, 'B_RepurchasedShares': 6, 'I_RepurchasedShares': 7,
           'B_ClosingDate': 8, 'I_ClosingDate': 9, 'B_RepurchaseAmount': 10, 'I_RepurchaseAmount': 11},
    'eu': {'B_EquityHolder': 0, 'I_EquityHolder': 1, 'B_TradedShares': 2, 'I_TradedShares': 3, 'B_StartDate': 4,
           'I_StartDate': 5, 'B_EndDate': 6, 'I_EndDate': 7, 'B_LaterHoldingShares': 8, 'I_LaterHoldingShares': 9,
           'B_AveragePrice': 10, 'I_AveragePrice': 11},
    'eo': {'B_EquityHolder': 0, 'I_EquityHolder': 1, 'B_TradedShares': 2, 'I_TradedShares': 3, 'B_StartDate': 4,
           'I_StartDate': 5, 'B_EndDate': 6, 'I_EndDate': 7, 'B_LaterHoldingShares': 8, 'I_LaterHoldingShares': 9,
           'B_AveragePrice': 10, 'I_AveragePrice': 11},
    'ep': {'B_Pledger': 0, 'I_Pledger': 1, 'B_PledgedShares': 2, 'I_PledgedShares': 3, 'B_Pledgee': 4, 'I_Pledgee': 5,
           'B_TotalHoldingShares': 6, 'I_TotalHoldingShares': 7, 'B_TotalHoldingRatio': 8, 'I_TotalHoldingRatio': 9,
           'B_TotalPledgedShares': 10, 'I_TotalPledgedShares': 11, 'B_StartDate': 12, 'I_StartDate': 13,
           'B_EndDate': 14, 'I_EndDate': 15, 'B_ReleasedDate': 16, 'I_ReleasedDate': 17}}

train_event_nums = 34
test_event_nums = 16

def parsing_document(file,tree_file,data_type):
    stopwords = []
    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f_stopword:
        stopword_datas = f_stopword.readlines()
        for stopword in stopword_datas:
            stopwords.append(stopword.strip())

    ltp = LTP()
    with open(file, 'r', encoding='utf-8-sig') as fp:
        datas = json.load(fp)

    if data_type == 'train':
        event_num = train_event_nums
    elif data_type == 'test':
        event_num = test_event_nums
    else:
        event_num = None

    docs = []
    for doc in datas:
        sentences = doc[1]['sentences']
        events = doc[1]['recguid_eventname_eventdict_list']
        arguments = doc[1]['ann_mspan2dranges']
        doc_tree_obj = Doc_Tree()
        sen_id = 0
        word_id = 0
        for sentence in sentences:
            token_ids = []
            sentence = sentence.strip('')
            if len(sentence) == 0:
                continue
            words, hidden = ltp.seg([sentence.strip()])
            words = words[0]
            word_pos = 0
            word_pos_list = []
            for i, word in enumerate(words):
                word_pos_list.append([sen_id, word_pos, word_pos + len(word)])
                word_pos = word_pos + len(word)
                token_ids.append(i)
            token_labels = get_token_labels(words, word_pos_list, arguments, events, event_num)
            pos = ltp.pos(hidden)[0]
            deps = ltp.dep(hidden)[0]
            dep_new = []
            for j,dep in enumerate(deps):
                if dep[1] == 0:
                    dep_new.append((dep[0] + word_id, dep[1], dep[2]))
                else:
                    dep_new.append((dep[0] + word_id, dep[1] + word_id, dep[2]))
            doc_tree_obj.build_dp_tree_ltp4(words, dep_new, pos, DROOT, sen_id, word_pos_list, token_ids,token_labels)
            doc_tree_obj.remove_stop_word_nodes_tree(stopwords)
            sen_id += 1
            word_id += len(words)
        docs.append(doc_tree_obj)

    with open(tree_file, 'wb') as f:
        pickle.dump(docs, f, -1)


def get_token_labels(words, word_pos_list, arguments, events, event_num):
    token_labels = []
    for i, token in enumerate(words):
        labels = init_role_event_labels(event_num)
        for argument, position_list in arguments.items():
            if token in argument:
                for position in position_list:
                    if word_pos_list[i][0] == position[0] and word_pos_list[i][1] == position[1] and word_pos_list[i][
                        2] <= position[2]:
                        update_token_role_event(token, events, 'B_', labels, event_num)
                    elif word_pos_list[i][0] == position[0] and word_pos_list[i][1] > position[1] and word_pos_list[i][
                        1] < position[2]:
                        update_token_role_event(token, events, 'I_', labels, event_num)
        token_labels.append(labels)
    return token_labels


def init_role_event_labels(event_num):
    labels = {}
    for value in event_type_f2s.values():
        role_event = np.zeros([event_num])
        labels[value] = role_event
    return labels


def update_token_role_event(word, events, role_subtype, labels, event_num):
    event_ids = {'ef': 0, 'er': 0, 'eu': 0, 'eo': 0, 'ep': 0}
    for i, event in enumerate(events):
        event_type = event_type_f2s[event[1]]
        event_arguments = event[2]
        for role, value in event_arguments.items():
            if value is None:
                continue
            if event_ids[event_type] > event_num - 1:
                continue
            if word in value:
                role_id = event_roles[event_type][role_subtype + role]
                labels[event_type][event_ids[event_type]] = role_id + 1
                event_ids[event_type] += 1


if __name__ == '__main__':
    parsing_document('./data/train.json','./data/train.pkl','train')
    parsing_document('./data/test.json','./data/test.pkl','test')
