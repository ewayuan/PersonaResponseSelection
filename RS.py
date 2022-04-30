import argparse
import itertools
import json
import multiprocessing as mp
import os
import pickle
import random
import re
import string
import sys
import time
import json
import math
import copy
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup

from util import load_pickle, save_pickle, count_parameters, compute_metrics, compute_metrics_from_logits
from functions import *

from gensim.models import LdaModel
from pprint import pprint
from contextlib import redirect_stdout
from nltk.corpus import stopwords
from gensim.test.utils import datapath

import logging

logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")



def main(config, progress):
    # save config
    with open("./log/configs.json", "a") as f:
        json.dump(config, f)
        f.write("\n")
    cprint("*"*80)
    cprint("Experiment progress: {0:.2f}%".format(progress*100))
    cprint("*"*80)
    metrics = {}

    # data hyper-params
    train_path = config["train_path"]
    valid_path = config["valid_path"]
    test_path = config["test_path"]
    dataset = train_path.split("/")[3]
    test_mode = bool(config["test_mode"])
    load_model_path = config["load_model_path"]
    save_model_path = config["save_model_path"]
    num_candidates = config["num_candidates"]
    num_personas = config["num_personas"]
    persona_path = config["persona_path"]
    max_sent_len = config["max_sent_len"]
    max_seq_len = config["max_seq_len"]
    PEC_ratio = config["PEC_ratio"]
    train_ratio = config["train_ratio"]
    if PEC_ratio != 0 and train_ratio != 1:
        raise ValueError("PEC_ratio or train_ratio not qualified!")

    # model hyper-params
    config_id = config["config_id"]
    model = config["model"]
    shared = bool(config["shared"])
    apply_interaction = bool(config["apply_interaction"])
    matching_method = config["matching_method"]
    aggregation_method = config["aggregation_method"]
    output_hidden_states = False

    # training hyper-params
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    warmup_steps = config["warmup_steps"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    lr = config["lr"]
    weight_decay = 0
    seed = config["seed"]
    device = torch.device(config["device"])
    fp16 = bool(config["fp16"])
    fp16_opt_level = config["fp16_opt_level"]

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if test_mode and load_model_path == "":
        raise ValueError("Must specify test model path when in test mode!")

    # load data
    cprint("Loading conversation data...")
    train = load_pickle(train_path)
    valid = load_pickle(valid_path)
    if test_mode:
        test = load_pickle(test_path)
        valid_path = test_path
        valid = test

    cprint("sample train data: ", train[0])
    cprint("sample valid data: ", valid[0])

    # Tring to create train, valid, test DataLoader to group datas
    # 1. fistly, train, valid, test DataLoader to calcualate word level embedding
    # 2. and then, sentence embedding
    # 3. In the end, topic level embedding
    cprint ("Creating Word Level Embedding ...")
    # tokenization
    cprint("Tokenizing ...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cached_tokenized_train_path = train_path.replace(".pkl", "_tokenized.pkl")
    cached_tokenized_valid_path = valid_path.replace(".pkl", "_tokenized.pkl")
    if os.path.exists(cached_tokenized_train_path):
        cprint("Loading tokenized dataset from ", cached_tokenized_train_path)
        word_level_train = load_pickle(cached_tokenized_train_path)
        cprint("after tokenize_conversations: ", word_level_train[99])
    else:
        word_level_train = tokenize_conversations(train, tokenizer, max_sent_len)
        cprint("after tokenize_conversations: ", word_level_train[99])
        cprint("Saving tokenized dataset to ", cached_tokenized_train_path)
        save_pickle(word_level_train, cached_tokenized_train_path)

    if os.path.exists(cached_tokenized_valid_path):
        cprint("Loading tokenized dataset from ", cached_tokenized_valid_path)
        word_level_valid = load_pickle(cached_tokenized_valid_path)
    else:
        word_level_valid = tokenize_conversations(valid, tokenizer, max_sent_len)
        cprint("Saving tokenized dataset to ", cached_tokenized_valid_path)
        save_pickle(word_level_valid, cached_tokenized_valid_path)

    persona = None
    if num_personas > 0:
        cprint("Tokenizing persona sentences...")
        cached_tokenized_persona_path = persona_path.replace(".pkl", "_tokenized.pkl")
        if os.path.exists(cached_tokenized_persona_path):
            cprint("Loading tokenized persona from file...")
            persona = load_pickle(persona_path)
            word_level_persona = load_pickle(cached_tokenized_persona_path)
        else:
            cprint("Loading persona data...")
            persona = load_pickle(persona_path)
            all_speakers = set([s for conv in load_pickle(config["train_path"]) + \
                load_pickle(config["valid_path"]) + load_pickle(config["test_path"]) for s, sent in conv])
            cprint("Tokenizing persona data...")
            word_level_persona = tokenize_personas(persona, tokenizer, all_speakers, num_personas)
            cprint("Saving tokenized persona to file...")
            save_pickle(word_level_persona, cached_tokenized_persona_path)
        cprint("Persona dataset statistics (after tokenization):", len(word_level_persona))
        cprint("Sample tokenized persona:", list(word_level_persona.values())[0])
    cprint("Sample tokenized data: ")
    cprint("train: ", word_level_train[0])
    cprint("test: ", word_level_valid[0])
    cprint("persona: ", list(word_level_persona.values())[0])
    # select subsets of training and validation data for casualconversation
    cprint("dataset: ", dataset)
    if dataset == "casualconversation_v3":
        cprint("reducing dataset size...")
        word_level_train = word_level_train[:150000]
        word_level_valid = word_level_valid[:20000]

    # if train_ratio != 1:
    #     num_train_examples = int(len(word_level_train) * train_ratio)
    #     cprint("reducing training set size to {0}...".format(num_train_examples))
    #     word_level_train = train[:num_train_examples]
    #
    # if PEC_ratio != 0:
    #     cprint("Replacing {0} of casual to PEC...".format(PEC_ratio))
    #     cprint(len(word_level_train))
    #
    #     PEC_train_path = "./data/reddit_empathetic/combined_v3/train_cleaned_bert.pkl"
    #     PEC_persona_path = "./data/reddit_empathetic/combined_v3/persona_10.pkl"
    #
    #     # load cached casual conversations and persona
    #     num_PEC_examples = int(len(train) * PEC_ratio)
    #     train[:num_PEC_examples] = load_pickle(PEC_train_path.replace(".pkl", "_tokenized.pkl"))[:num_PEC_examples]
    #     cprint(num_PEC_examples, len(train))
    #
    #     if num_personas > 0:
    #         cprint("number of speakers before merging PEC and casual: ", len(persona))
    #         # merge persona
    #         PEC_persona = load_pickle(PEC_persona_path.replace(".pkl", "_tokenized.pkl"))
    #         for k,v in PEC_persona.items():
    #             if k not in persona:
    #                 persona[k] = v
    #         cprint("number of speakers after merging PEC and casual: ", len(persona))

    # create context and response
    cprint("before create_context_and_response: ", len(word_level_train))
    word_level_train = create_context_and_response(word_level_train)
    word_level_valid = create_context_and_response(word_level_valid)
    cprint("Sample context and response: ")
    cprint(word_level_train[0])
    cprint("len(word_level_train): ", len(word_level_train))
    cprint(word_level_valid[0])
    cprint("len(word_level_valid): ", len(word_level_valid))

    # convert to token ids
    cprint("Converting conversations to ids: ")
    if not test_mode:
        all_context_ids_train, all_context_attention_mask_train, all_context_token_type_ids_train, \
        all_response_ids_train, all_response_attention_mask_train, all_response_token_type_ids_train, \
        all_persona_ids_train, all_persona_attention_mask_train, all_persona_token_type_ids_train = \
            convert_conversations_to_ids(word_level_train, word_level_persona, tokenizer, max_seq_len, max_sent_len, num_personas)
    # For valid
    all_context_ids_valid, all_context_attention_mask_valid, all_context_token_type_ids_valid, \
    all_response_ids_valid, all_response_attention_mask_valid, all_response_token_type_ids_valid, \
    all_persona_ids_valid, all_persona_attention_mask_valid, all_persona_token_type_ids_valid = \
        convert_conversations_to_ids(word_level_valid, word_level_persona, tokenizer, max_seq_len, max_sent_len, num_personas)


    # Topic Modelling
    # processing topic level persona
    all_speakers = set([s for conv in load_pickle(config["train_path"]) + \
                load_pickle(config["valid_path"]) + load_pickle(config["test_path"]) for s, sent in conv])
    topic_level_persona = merge_persona_token_to_sents (persona, all_speakers, num_personas)
    #
    cprint("Creating Topic Modelling ...")
    model_path = "model_100"
    lda = LdaModel.load(datapath(model_path))
    common_dict = load_pickle("./TopicModelling/dialog_data_dic.p")
    topic_embedding = load_pickle("./TopicModelling/topic_frequent_words_embedding.pkl")
    # train: (context, reponse, spearker)
    if not test_mode:
        cached_context_topic_distribution_train_path = train_path.replace(".pkl", "cached_context_topic_distribution_train.pkl")
        cached_response_topic_distribution_train_path = train_path.replace(".pkl", "cached_response_topic_distribution_train.pkl")
        cached_persona_topic_distribution_train_path = train_path.replace(".pkl", "cached_persona_topic_distribution_train.pkl")
        cprint("cached_response_topic_distribution_train_path: ", cached_response_topic_distribution_train_path)
        if os.path.exists(cached_context_topic_distribution_train_path) and os.path.exists(cached_response_topic_distribution_train_path) and os.path.exists(cached_persona_topic_distribution_train_path):
            cprint("Loading Context and Response Topic Modelling for train...")
            context_topic_distribution_train = load_pickle(cached_context_topic_distribution_train_path)
            response_topic_distribution_train = load_pickle(cached_response_topic_distribution_train_path)
            persona_topic_distribution_train = load_pickle(cached_persona_topic_distribution_train_path)
        else:
            cprint("Create Context and Response Topic Modelling for train...")
            topic_modelling_train = create_context_and_response_topic_modelling(train)
            cprint("topic_modelling_train: ", len(topic_modelling_train))
            # topic_modelling_train: [(topic_distribution_of_context: List, topic_distribution_of_response: List, speaker)]
            context_topic_distribution_train, response_topic_distribution_train, persona_topic_distribution_train = \
                generate_data_topic_distribuition(topic_modelling_train, topic_level_persona, lda, common_dict)
            save_pickle(context_topic_distribution_train, cached_context_topic_distribution_train_path)
            save_pickle(response_topic_distribution_train, cached_response_topic_distribution_train_path)
            save_pickle(persona_topic_distribution_train, cached_persona_topic_distribution_train_path)


        # cprint("Convert to Topic Embedding Data for trian")
        # # topic_embedding_train: [context_topic_embedding, response_topic_embedding, persona_topic_embedding]
        # # context_topic_embedding = [(topic_id, topic_embedding, topic_prob), ..., ...]
        # topic_embedding_train = convert_to_topic_embedding_data (topic_embedding, topic_modelling_train)
        # # topic_embedding_train/valid: TensorDataset:(all_Uce_context, all_UPct_context, all_Uce_response, all_UPct_response, all_Uce_persona, all_UPct_persona)
        # #                               all_Uce_context size:  torch.Size([n, 100, 768]) all_UPct_context size:  torch.Size([n, 1, 100])
        # #                               all_Uce_response size:  torch.Size([n, 100, 768]) all_UPct_response size:  torch.Size([n, 1, 100])
        # #                               all_Uce_persona size:  torch.Size([n, 100, 768]) all_UPct_persona size:  torch.Size([n, 1, 100])
        # cprint("Convert topic embdding matrix for train")
        # all_Uce_context_train, all_UPct_context_train, all_context_topic_mask_train,  \
        # all_Uce_response_train, all_UPct_response_train, all_response_topic_mask_train,\
        # all_Uce_persona_train, all_UPct_persona_train, all_persona_topic_mask_train = convert_topic_embedding_matrix(topic_embedding_train)

    cached_context_topic_distribution_valid_path = valid_path.replace(".pkl", "cached_context_topic_distribution_valid.pkl")
    cached_response_topic_distribution_valid_path = valid_path.replace(".pkl", "cached_response_topic_distribution_valid.pkl")
    cached_persona_topic_distribution_valid_path = valid_path.replace(".pkl", "cached_persona_topic_distribution_valid.pkl")
    if os.path.exists(cached_context_topic_distribution_valid_path) and os.path.exists(cached_response_topic_distribution_valid_path) and os.path.exists(cached_persona_topic_distribution_valid_path):
        cprint("Loading Context and Response Topic Modelling for valid...")
        context_topic_distribution_valid = load_pickle(cached_context_topic_distribution_valid_path)
        response_topic_distribution_valid = load_pickle(cached_response_topic_distribution_valid_path)
        persona_topic_distribution_valid = load_pickle(cached_persona_topic_distribution_valid_path)
    else:
        cprint("Create Context and Response Topic Modelling for valid...")
        topic_modelling_valid = create_context_and_response_topic_modelling(valid)
        # topic_modelling_valid: [(topic_distribution_of_context: List, topic_distribution_of_response: List, speaker)]
        context_topic_distribution_valid, response_topic_distribution_valid, persona_topic_distribution_valid = \
            generate_data_topic_distribuition(topic_modelling_valid, topic_level_persona, lda, common_dict)
        save_pickle(context_topic_distribution_valid, cached_context_topic_distribution_valid_path)
        save_pickle(response_topic_distribution_valid, cached_response_topic_distribution_valid_path)
        save_pickle(persona_topic_distribution_valid, cached_persona_topic_distribution_valid_path)

    # cprint("Convert to Topic Embedding Data for valid")
    # topic_embedding_valid = convert_to_topic_embedding_data (topic_embedding, topic_modelling_valid)
    # # topic_embedding_valid: [context_topic_embedding, response_topic_embedding, persona_topic_embedding]
    # # context_topic_embedding = [(topic_id, topic_embedding, topic_prob), ..., ...]
    # cprint("Convert spearker to persona Topic Embedding for valid")
    # all_Uce_context_valid, all_UPct_context_valid, all_context_topic_mask_valid, \
    # all_Uce_response_valid, all_UPct_response_valid, all_response_topic_mask_valid, \
    # all_Uce_persona_valid, all_UPct_persona_valid, all_persona_topic_mask_valid = convert_topic_embedding_matrix(topic_embedding_valid)


    # Create word_level_train Dataloader & topic_embedding_train Dataloader

    cprint("Create word_level_train Dataloader & topic_embedding_train Dataloader...")
    cprint("all_context_ids_valid: ", all_context_ids_valid.size())
    # cprint("all_context_attention_mask_valid: ", all_context_attention_mask_valid.size())
    # cprint("all_context_token_type_ids_valid: ", all_context_token_type_ids_valid.size())
    # cprint("all_persona_ids_valid: ", all_persona_ids_valid.size())
    # cprint("all_persona_attention_mask_valid: ", all_persona_attention_mask_valid.size())
    # cprint("all_persona_token_type_ids_valid: ", all_persona_token_type_ids_valid.size())
    # cprint("all_response_ids_valid: ", all_response_ids_valid.size())
    # cprint("all_response_attention_mask_valid: ", all_response_attention_mask_valid.size())
    # cprint("all_response_token_type_ids_valid: ", all_response_token_type_ids_valid.size())
    # cprint("all_Uce_context_valid: ", all_Uce_context_valid.size())
    # cprint("all_UPct_context_valid: ", all_UPct_context_valid.size())
    # cprint("all_Uce_response_valid: ", all_Uce_response_valid.size())
    # cprint("all_UPct_response_valid: ", all_UPct_response_valid.size())
    # cprint("all_Uce_persona_valid: ", all_Uce_persona_valid.size())
    # cprint("all_UPct_persona_valid: ", all_UPct_persona_valid.size())
    # cprint("all_context_topic_mask_valid: ", all_context_topic_mask_valid.size())
    # cprint("all_response_topic_mask_valid: ", all_response_topic_mask_valid.size())
    # cprint("all_persona_topic_mask_valid: ", all_persona_topic_mask_valid.size())
    cprint("all_context_ids_train: ", all_context_ids_train.size())

    cprint("all_context_attention_mask_train: ", all_context_attention_mask_train.size())
    cprint("all_context_token_type_ids_train: ", all_context_token_type_ids_train.size())
    cprint("all_response_ids_train: ", all_response_ids_train.size())
    cprint("all_response_attention_mask_train: ", all_response_attention_mask_train.size())
    cprint("all_response_token_type_ids_train: ", all_response_token_type_ids_train.size())
    cprint("all_persona_ids_train: ", all_persona_ids_train.size())
    cprint("all_persona_attention_mask_train: ", all_persona_attention_mask_train.size())
    cprint("all_persona_token_type_ids_train: ", all_persona_token_type_ids_train.size())

    cprint("context_topic_distribution_train: ", context_topic_distribution_train.size())
    cprint("response_topic_distribution_train: ", response_topic_distribution_train.size())
    cprint("persona_topic_distribution_train: ", persona_topic_distribution_train.size())

    cprint("all_context_ids_valid: ", all_context_ids_valid.size())
    cprint("all_context_attention_mask_valid: ", all_context_attention_mask_valid.size())
    cprint("all_context_token_type_ids_valid: ", all_context_token_type_ids_valid.size())
    cprint("all_response_ids_valid: ", all_response_ids_valid.size())
    cprint("all_response_attention_mask_valid: ", all_response_attention_mask_valid.size())
    cprint("all_response_token_type_ids_valid: ", all_response_token_type_ids_valid.size())
    cprint("all_persona_ids_valid: ", all_persona_ids_valid.size())
    cprint("all_persona_attention_mask_valid: ", all_persona_attention_mask_valid.size())
    cprint("all_persona_token_type_ids_valid: ", all_persona_token_type_ids_valid.size())
    cprint("context_topic_distribution_valid: ", context_topic_distribution_valid.size())
    cprint("response_topic_distribution_valid: ", response_topic_distribution_valid.size())
    cprint("persona_topic_distribution_valid: ", persona_topic_distribution_valid.size())

    if not test_mode:
        train_dataset = TensorDataset(all_context_ids_train, all_context_attention_mask_train, all_context_token_type_ids_train, \
                                      all_response_ids_train, all_response_attention_mask_train, all_response_token_type_ids_train, \
                                      all_persona_ids_train, all_persona_attention_mask_train, all_persona_token_type_ids_train,\
                                      context_topic_distribution_train, response_topic_distribution_train, persona_topic_distribution_train)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)
        t_total = len(train_dataloader) // gradient_accumulation_steps * epochs

    valid_dataset = TensorDataset(all_context_ids_valid, all_context_attention_mask_valid, all_context_token_type_ids_valid, \
                                  all_response_ids_valid, all_response_attention_mask_valid, all_response_token_type_ids_valid, \
                                  all_persona_ids_valid, all_persona_attention_mask_valid, all_persona_token_type_ids_valid,\
                                  context_topic_distribution_valid, response_topic_distribution_valid, persona_topic_distribution_valid)
    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=num_candidates)


    # Create model
    cprint("Building dmodel...")
    bert_model = BertModel.from_pretrained(model, output_hidden_states=output_hidden_states).to(device)

    model = ourModel(device, args=args)
    cprint(model)
    cprint("number of parameters: ", count_parameters(model))

    if shared: # responses, persona, context share same bert, else, they can have their own bert models.
        cprint("number of encoders: 1")
        models = [model]
    else:
        if num_personas == 0:
            cprint("number of encoders: 2")
            # models = [model, copy.deepcopy(model)]
            models = [model, pickle.loads(pickle.dumps(model))]
        else:
            cprint("number of encoders: 3")
            # models = [model, copy.deepcopy(model), copy.deepcopy(model)]
            models = [model, pickle.loads(pickle.dumps(model)), pickle.loads(pickle.dumps(model))]

    if test_mode:
        cprint("Loading weights from ", load_model_path)
        model.load_state_dict(torch.load(load_model_path))
        models = [model]

    for i, model in enumerate(models):
        cprint("model {0} number of parameters: ".format(i), count_parameters(model))
        model.to(device)

    # optimization
    amp = None
    if fp16:
        from apex import amp

    no_decay = ["bias", "LayerNorm.weight"]
    optimizers = []
    schedulers = []
    for i, model in enumerate(models):
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": weight_decay,
        #     },
        #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        # ]
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=0)

        if fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
            # models[i] = nn.DataParallel(model, device_ids=[0,1,2])
        optimizers.append(optimizer)

        if not test_mode:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
            schedulers.append(scheduler)

    if test_mode:
        # evaluation
        for model in models:
            model.eval()
        valid_iterator = tqdm(valid_dataloader, desc="Iteration")
        valid_loss, (valid_acc, valid_recall, valid_MRR) = evaluate_epoch(valid_iterator, models, \
            num_personas, gradient_accumulation_steps, device, dataset, 0, apply_interaction, matching_method, aggregation_method, bert_model)
        cprint("test loss: {0:.4f}, test acc: {1:.4f}, test recall: {2}, test MRR: {3:.4f}"
            .format(valid_loss, valid_acc, valid_recall, valid_MRR))
        sys.exit()

    # training
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_valid_accs = []
    epoch_valid_recalls = []
    epoch_valid_MRRs = []
    cprint("***** Running training *****")
    cprint("Num examples =", len(train_dataset))
    cprint("Num Epochs =", epochs)
    cprint("Total optimization steps =", t_total)
    best_model_statedict = {}
    for epoch in range(epochs):
        cprint("Epoch", epoch+1)
        # training
        for model in models:
            model.train()
        train_iterator = tqdm(train_dataloader, desc="Iteration")

        train_loss, (train_acc, _, _) = train_epoch(train_iterator, models, num_personas, optimizers, \
            schedulers, gradient_accumulation_steps, device, fp16, amp, apply_interaction, matching_method, aggregation_method, topic_embedding, bert_model)
        epoch_train_losses.append(train_loss)

        # evaluation
        for model in models:
            model.eval()
        valid_iterator = tqdm(valid_dataloader, desc="Iteration")
        valid_loss, (valid_acc, valid_recall, valid_MRR) = evaluate_epoch(valid_iterator, models, \
            num_personas, gradient_accumulation_steps, device, dataset, epoch, apply_interaction, matching_method, aggregation_method, topic_embedding, bert_model)

        cprint("Config id: {7}, Epoch {0}: train loss: {1:.4f}, valid loss: {2:.4f}, train_acc: {3:.4f}, valid acc: {4:.4f}, valid recall: {5}, valid_MRR: {6:.4f}"
            .format(epoch+1, train_loss, valid_loss, train_acc, valid_acc, valid_recall, valid_MRR, config_id))

        epoch_valid_losses.append(valid_loss)
        epoch_valid_accs.append(valid_acc)
        epoch_valid_recalls.append(valid_recall)
        epoch_valid_MRRs.append(valid_MRR)

        if save_model_path != "":
            if epoch == 0:
                for k, v in models[0].state_dict().items():
                    best_model_statedict[k] = v.cpu()
            else:
                if epoch_valid_recalls[-1][0] == max([recall1 for recall1, _, _ in epoch_valid_recalls]):
                    for k, v in models[0].state_dict().items():
                        best_model_statedict[k] = v.cpu()


    config.pop("seed")
    config.pop("config_id")
    metrics["config"] = config
    metrics["score"] = max(epoch_valid_accs)
    metrics["epoch"] = np.argmax(epoch_valid_accs).item()
    metrics["recall"] = epoch_valid_recalls
    metrics["MRR"] = epoch_valid_MRRs

    if save_model_path:
        cprint("Saving model to ", save_model_path)
        # cprint("best_model_statedict: ", best_model_statedict)
        torch.save(best_model_statedict, save_model_path)

    return metrics


def clean_config(configs):
    cleaned_configs = []
    for config in configs:
        if config not in cleaned_configs:
            cleaned_configs.append(config)
    return cleaned_configs


def merge_metrics(metrics):
    avg_metrics = {"score" : 0}
    num_metrics = len(metrics)
    for metric in metrics:
        for k in metric:
            if k != "config":
                avg_metrics[k] += np.array(metric[k])

    for k, v in avg_metrics.items():
        avg_metrics[k] = (v/num_metrics).tolist()

    return avg_metrics


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Model for Transformer-based Dialogue Generation with Controlled Emotion")
    parser.add_argument('--config', help='Config to read details', required=True)
    parser.add_argument('--note', help='Experiment note', default="")
    args = parser.parse_args()
    cprint("Experiment note: ", args.note)
    with open(args.config) as configfile:
        config = json.load(configfile) # config is now a python dict

    # pass experiment config to main
    parameters_to_search = OrderedDict() # keep keys in order
    other_parameters = {}
    keys_to_omit = ["kernel_sizes"] # keys that allow a list of values
    for k, v in config.items():
        # if value is a list provided that key is not device, or kernel_sizes is a nested list
        if isinstance(v, list) and k not in keys_to_omit:
            parameters_to_search[k] = v
        elif k in keys_to_omit and isinstance(config[k], list) and isinstance(config[k][0], list):
            parameters_to_search[k] = v
        else:
            other_parameters[k] = v

    if len(parameters_to_search) == 0:
        config_id = time.perf_counter()
        config["config_id"] = config_id
        cprint(config)
        output = main(config, progress=1)
        cprint("-"*80)
        cprint("config: ", output["config"])
        cprint("epoch: ", output["epoch"])
        cprint("score: ", output["score"])
        cprint("recall: ", output["recall"])
        cprint("MRR: ", output["MRR"])
    else:
        all_configs = []
        for i, r in enumerate(itertools.product(*parameters_to_search.values())):
            specific_config = {}
            for idx, k in enumerate(parameters_to_search.keys()):
                specific_config[k] = r[idx]

            # merge with other parameters
            merged_config = {**other_parameters, **specific_config}
            all_configs.append(merged_config)

        #   cprint all configs
        for config in all_configs:
            config_id = time.perf_counter()
            config["config_id"] = config_id
            logging.critical("config id: {0}".format(config_id))
            cprint(config)
            cprint("\n")

        # multiprocessing
        num_configs = len(all_configs)
        # mp.set_start_method('spawn')
        pool = mp.Pool(processes=config["processes"])
        results = [pool.apply_async(main, args=(x,i/num_configs)) for i,x in enumerate(all_configs)]
        outputs = [p.get() for p in results]

        # if run multiple models using different seed and get the averaged result
        if "seed" in parameters_to_search:
            all_metrics = []
            all_cleaned_configs = clean_config([output["config"] for output in outputs])
            for config in all_cleaned_configs:
                metrics_per_config = []
                for output in outputs:
                    if output["config"] == config:
                        metrics_per_config.append(output)
                avg_metrics = merge_metrics(metrics_per_config)
                all_metrics.append((config, avg_metrics))
            # log metrics
            cprint("Average evaluation result across different seeds: ")
            for config, metric in all_metrics:
                cprint("-"*80)
                cprint(config)
                cprint(metric)

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for config, metric in all_metrics:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(config) + "\n")
                    f.write(json.dumps(metric) + "\n")

        else:
            for output in outputs:
                cprint("-"*80)
                cprint(output["config"])
                cprint(output["score"])
                cprint(output["recall"])
                cprint(output["MRR"])
                cprint("Best result at epoch {0}: ".format(output["epoch"]))
                cprint(output["recall"][output["epoch"]], output["MRR"][output["epoch"]])

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for output in outputs:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(output) + "\n")
