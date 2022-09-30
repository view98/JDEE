import logging
import random
import numpy as np
import torch
import gc
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as prf
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange

from datasets import my_collate
torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_input_from_batch(batch):
    inputs = { 'token_ids':batch[0],
               'pos_ids':batch[1],
               'dep_ids':batch[2],
               'sen_pos_ids':batch[3],
               'word_pos_ids':batch[4],
               'parent_ids':batch[5],
               'ppos_ids':batch[6],
               'pdep_ids':batch[7],
               'event_ids':batch[8]
                }
    ef_labels = batch[9]
    er_labels = batch[10]
    eu_labels = batch[11]
    eo_labels = batch[12]
    ep_labels = batch[13]

    return inputs, ef_labels, er_labels, eu_labels, eo_labels, ep_labels


def get_collate_fn():
    return my_collate

def train(args,model,train_dataset,test_dataset,train_labels_weight,test_labels_weight):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn()
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    all_ef_results = []
    all_er_results = []
    all_eu_results = []
    all_eo_results = []
    all_ep_results = []
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epoch = 0
    ef_weight = torch.from_numpy(np.array(train_labels_weight[:17],dtype=np.float32)).to(args.device)
    er_weight = torch.from_numpy(np.array(train_labels_weight[17:30],dtype=np.float32)).to(args.device)
    eu_weight = torch.from_numpy(np.array(train_labels_weight[30:43],dtype=np.float32)).to(args.device)
    eo_weight = torch.from_numpy(np.array(train_labels_weight[43:56],dtype=np.float32)).to(args.device)
    ep_weight = torch.from_numpy(np.array(train_labels_weight[56:],dtype=np.float32)).to(args.device)
    f = open('./output/result.txt','w',encoding='utf-8')
    for _ in train_iterator:
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs, ef_labels,er_labels,eu_labels,eo_labels,ep_labels = get_input_from_batch(batch)
            ef_logit, er_logit, eu_logit, eo_logit, ep_logit = model(**inputs)

            ef_role_loss = F.cross_entropy(ef_logit, ef_labels,weight=ef_weight)
            er_role_loss = F.cross_entropy(er_logit, er_labels,weight=er_weight)
            eu_role_loss = F.cross_entropy(eu_logit, eu_labels,weight=eu_weight)
            eo_role_loss = F.cross_entropy(eo_logit, eo_labels,weight=eo_weight)
            ep_role_loss = F.cross_entropy(ep_logit, ep_labels,weight=ep_weight)

            loss = ef_role_loss+er_role_loss+eu_role_loss+eo_role_loss+ep_role_loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("  train_loss: %s", str((tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

        epoch += 1
        # ef_results, er_results, eu_results, eo_results, ep_results, eval_loss = evaluate(args, test_dataset, model,test_labels_weight,f)
        # all_ef_results.append(ef_results)
        # all_er_results.append(er_results)
        # all_eu_results.append(eu_results)
        # all_eo_results.append(eo_results)
        # all_ep_results.append(ep_results)
        tb_writer.add_scalar('train_epoch_loss',(tr_loss - logging_loss) / args.logging_steps, epoch)


    tb_writer.close()
    torch.save(model, './output/train.model')
    ef_results, er_results, eu_results, eo_results, ep_results, eval_loss = evaluate(args, test_dataset,model,test_labels_weight,f)
    all_ef_results.append(ef_results)
    all_er_results.append(er_results)
    all_eu_results.append(eu_results)
    all_eo_results.append(eo_results)
    all_ep_results.append(ep_results)
    return global_step, tr_loss/global_step, all_ef_results, all_er_results, all_eu_results, all_eo_results, all_ep_results

def evaluate(args, eval_dataset, model,test_labels_weight,f):
    ef_results = {}
    er_results = {}
    eu_results = {}
    eo_results = {}
    ep_results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn()
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,collate_fn=collate_fn)
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    ef_preds = None
    er_preds = None
    eu_preds = None
    eo_preds = None
    ep_preds = None
    out_ef_label_ids = None
    out_er_label_ids = None
    out_eu_label_ids = None
    out_eo_label_ids = None
    out_ep_label_ids = None

    ef_weight = torch.from_numpy(np.array(test_labels_weight[:17],dtype=np.float32)).to(args.device)
    er_weight = torch.from_numpy(np.array(test_labels_weight[17:30],dtype=np.float32)).to(args.device)
    eu_weight = torch.from_numpy(np.array(test_labels_weight[30:43],dtype=np.float32)).to(args.device)
    eo_weight = torch.from_numpy(np.array(test_labels_weight[43:56],dtype=np.float32)).to(args.device)
    ep_weight = torch.from_numpy(np.array(test_labels_weight[56:],dtype=np.float32)).to(args.device)

    with torch.no_grad():
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            inputs, ef_labels,er_labels,eu_labels,eo_labels,ep_labels = get_input_from_batch(batch)

            ef_logit, er_logit, eu_logit, eo_logit, ep_logit = model(**inputs)

            ef_loss = F.cross_entropy(ef_logit, ef_labels,weight=ef_weight)
            er_loss = F.cross_entropy(er_logit, er_labels,weight=er_weight)
            eu_loss = F.cross_entropy(eu_logit, eu_labels,weight=eu_weight)
            eo_loss = F.cross_entropy(eo_logit, eo_labels,weight=eo_weight)
            ep_loss = F.cross_entropy(ep_logit, ep_labels,weight=ep_weight)

            loss = ef_loss + er_loss + eu_loss + eo_loss + ep_loss
            tmp_eval_loss = loss
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if ef_preds is None:
                ef_preds = ef_logit.detach().cpu().numpy()
                er_preds = er_logit.detach().cpu().numpy()
                eu_preds = eu_logit.detach().cpu().numpy()
                eo_preds = eo_logit.detach().cpu().numpy()
                ep_preds = ep_logit.detach().cpu().numpy()

                out_ef_label_ids = ef_labels.detach().cpu().numpy()
                out_er_label_ids = er_labels.detach().cpu().numpy()
                out_eu_label_ids = eu_labels.detach().cpu().numpy()
                out_eo_label_ids = eo_labels.detach().cpu().numpy()
                out_ep_label_ids = ep_labels.detach().cpu().numpy()
            else:
                ef_preds = np.append(ef_preds, ef_logit.detach().cpu().numpy(), axis=0)
                er_preds = np.append(er_preds, er_logit.detach().cpu().numpy(), axis=0)
                eu_preds = np.append(eu_preds, eu_logit.detach().cpu().numpy(), axis=0)
                eo_preds = np.append(eo_preds, eo_logit.detach().cpu().numpy(), axis=0)
                ep_preds = np.append(ep_preds, ep_logit.detach().cpu().numpy(), axis=0)

                out_ef_label_ids = np.append(out_ef_label_ids, ef_labels.detach().cpu().numpy(), axis=0)
                out_er_label_ids = np.append(out_er_label_ids, er_labels.detach().cpu().numpy(), axis=0)
                out_eu_label_ids = np.append(out_eu_label_ids, eu_labels.detach().cpu().numpy(), axis=0)
                out_eo_label_ids = np.append(out_eo_label_ids, eo_labels.detach().cpu().numpy(), axis=0)
                out_ep_label_ids = np.append(out_ep_label_ids, ep_labels.detach().cpu().numpy(), axis=0)

            gc.collect()
            del batch,inputs,ef_labels,er_labels,eu_labels,eo_labels,ep_labels
            gc.collect()
            torch.cuda.empty_cache()

    ef_preds = np.argmax(ef_preds, axis=1)
    er_preds = np.argmax(er_preds, axis=1)
    eu_preds = np.argmax(eu_preds, axis=1)
    eo_preds = np.argmax(eo_preds, axis=1)
    ep_preds = np.argmax(ep_preds, axis=1)


    eval_loss = eval_loss / nb_eval_steps

    ef_result = prf_comput(ef_preds, out_ef_label_ids)
    ef_results.update(ef_result)
    er_result = prf_comput(er_preds, out_er_label_ids)

    er_results.update(er_result)
    eu_result = prf_comput(eu_preds, out_eu_label_ids)
    eu_results.update(eu_result)
    eo_result = prf_comput(eo_preds, out_eo_label_ids)
    # eo_result = compute_metrics(eo_preds, eo_trues)
    eo_results.update(eo_result)
    ep_result = prf_comput(ep_preds, out_ep_label_ids)
    ep_results.update(ep_result)


    logger.info('***** Eval results *****')
    logger.info(" eval loss: %s", str(eval_loss))
    logger.info("************ef*************")
    for key in ef_result.keys():
        logger.info("  %s = %s", key, str(ef_result[key]))
        f.write('ef'+key+'='+str(ef_result[key])+'\n')
    logger.info("************er*************")
    for key in er_result.keys():
        logger.info("  %s = %s", key, str(er_result[key]))
        f.write('er'+key+'='+str(er_result[key])+'\n')
    logger.info("************eu*************")
    for key in eu_result.keys():
        logger.info("  %s = %s", key, str(eu_result[key]))
        f.write('eu'+key + '=' + str(eu_result[key])+'\n')
    logger.info("************eo*************")
    for key in eo_result.keys():
        logger.info("  %s = %s", key, str(eo_result[key]))
        f.write('eo'+key + '=' + str(eo_result[key])+'\n')
    logger.info("************ep*************")
    for key in ep_result.keys():
        logger.info("  %s = %s", key, str(ep_result[key]))
        f.write('ep'+key + '=' + str(ep_result[key])+'\n')

    return ef_results,er_results,eu_results,eo_results,ep_results,eval_loss

def prf_comput(preds,labels):
    role_preds = []
    role_labels = []
    for i, label in enumerate(labels):
        if label > 0:
            role_preds.append(preds[i])
            role_labels.append(labels[i])

    micro_pre, micro_recall, micro_f1, micro_support = prf(y_true=role_labels, y_pred=role_preds, average='micro')
    return {
        "micro_pre": micro_pre,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1
    }
