import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup

import GPUtil
from pytorch_memlab import MemReporter

from util import load_pickle, save_pickle, count_parameters, compute_metrics, compute_metrics_from_logits

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
from contextlib import redirect_stdout
from nltk.corpus import stopwords
from gensim.test.utils import datapath
import torch.nn.init as init

import logging

class TransformerBlock(nn.Module):

    def __init__(self, input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, Q, K, V, mask=None, dropout=None, episilon=1e-8):
        '''
        :param Q: (batch_size, max_r_words, embedding_dim)
        :param K: (batch_size, max_u_words, embedding_dim)
        :param V: (batch_size, max_u_words, embedding_dim)
        :return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        '''
        dk = torch.Tensor([max(1.0, Q.size(-1))]).cuda()

        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        if mask is not None:
            Q_K = Q_K.masked_fill_(mask, -1e9)
        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_r_words, max_u_words)
        if dropout is not None:
            Q_K_score = dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            X = self.linear2(self.relu(self.linear1(X))) + X

        return X



class cnnBlock(nn.Module):
    def __init__(self):
        super(cnnBlock, self).__init__()
        self.cnn_2d_context_response_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.cnn_2d_context_response_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.maxpooling_context_response_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.affine_context_response = nn.Linear(in_features=6*62*1, out_features=200)
        self.relu = nn.ReLU()

        self.cnn_2d_persona_response_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.cnn_2d_persona_response_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.maxpooling_persona_response_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.affine_persona_response = nn.Linear(in_features=6*56*1, out_features=200)

        self.final_cnn_2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))
        self.final_maxpooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.affine_out = nn.Linear(in_features=99*3, out_features=1)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.cnn_2d_context_response_1.weight)
        init.xavier_normal_(self.cnn_2d_context_response_2.weight)
        init.xavier_normal_(self.affine_context_response.weight)

        init.xavier_normal_(self.cnn_2d_persona_response_1.weight)
        init.xavier_normal_(self.cnn_2d_persona_response_2.weight)
        init.xavier_normal_(self.affine_persona_response.weight)

        init.xavier_normal_(self.final_cnn_2d.weight)

        init.xavier_normal_(self.affine_out.weight)

    def cnn_contxt_response(self, matrix):
        matrix = matrix.unsqueeze(1)
        Z = self.cnn_2d_context_response_1(matrix)
        Z = self.maxpooling_context_response_1(Z)
        Z = self.relu(Z)

        Z = self.cnn_2d_context_response_2(Z)
        Z = self.maxpooling_context_response_1(Z)
        Z = self.relu(Z)

        Z = Z.view(Z.size(0), -1)

        Z = self.affine_context_response(Z)

        return Z

    def cnn_persona_response(self, matrix):
        matrix = matrix.unsqueeze(1)
        Z = self.cnn_2d_persona_response_1(matrix)
        Z = self.maxpooling_persona_response_1(Z)
        Z = self.relu(Z)

        Z = self.cnn_2d_persona_response_2(Z)
        Z = self.maxpooling_persona_response_1(Z)
        Z = self.relu(Z)

        Z = Z.view(Z.size(0), -1)

        Z = self.affine_persona_response(Z)

        return Z

    def forward(self, context_response_attn_similarity_matrix, context_response_similarity_matrix,\
                persona_response_attn_similarity_matrix, persona_response_similarity_matrix,\
                context_response_word_level_attn_simialrity_matrix, persona_response_word_level_attn_simialrity_matrix,\
                context_response_topic_level_attn_simialrity_matrix, persona_response_topic_level_attn_simialrity_matrix):

        context_response_attn_V = self.cnn_contxt_response(context_response_attn_similarity_matrix)
        context_response_V = self.cnn_contxt_response(context_response_similarity_matrix)
        persona_response_attn_V = self.cnn_persona_response(persona_response_attn_similarity_matrix)
        persona_response_V = self.cnn_persona_response(persona_response_similarity_matrix)
        context_response_word_level_attn_simialrity_matrix_V = self.cnn_contxt_response(context_response_word_level_attn_simialrity_matrix)
        persona_response_word_level_attn_simialrity_matrix_V = self.cnn_persona_response(persona_response_word_level_attn_simialrity_matrix)
        context_response_topic_level_attn_simialrity_matrix_V = self.cnn_contxt_response(context_response_topic_level_attn_simialrity_matrix)
        persona_response_topic_level_attn_simialrity_matrix_V = self.cnn_persona_response(persona_response_topic_level_attn_simialrity_matrix)
        stacked = torch.stack([context_response_attn_V, context_response_V, \
                             persona_response_attn_V, persona_response_V, \
                             context_response_word_level_attn_simialrity_matrix_V, persona_response_word_level_attn_simialrity_matrix_V,\
                             context_response_topic_level_attn_simialrity_matrix_V, persona_response_topic_level_attn_simialrity_matrix_V], dim=1).unsqueeze(1)
        output_stacked = self.final_cnn_2d(stacked)
        output_stacked = self.final_maxpooling(output_stacked)
        output_stacked = output_stacked.view(output_stacked.size(0), -1)
        matching_output = self.affine_out(output_stacked).squeeze()

        return matching_output

class ourModel (nn.Module):
    def __init__(self, device):
        super(ourModel, self).__init__()
        self.dialog_data_topic = load_pickle("./TopicModelling/dialog_data_topics.pkl")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model =  BertModel.from_pretrained("bert-base-uncased", output_hidden_states=False).to(device)
        self.embed_dim = 768
        self.response_len = 32
        self.context_len = 256
        self.persona_len = 231
        self.context_transformer  = TransformerBlock(input_size=self.embed_dim)
        self.response_transformer = TransformerBlock(input_size=self.embed_dim)
        self.persona_transformer = TransformerBlock(input_size=self.embed_dim)
        self.cnn_block = cnnBlock()

    def convert_topic_id_to_embedding(self, data, topic_embedding):
        # data: [(topic_id_1, topic_prob_1), (topic_id_2, topic_prob_2), ...]
        topic_embedding_prob_list = []
        for topic_id, topic_prob in data:
            if int(topic_id) == 0 and topic_prob == 0:
                continue
            topic_embedding_prob_list.append((topic_id, topic_embedding[int(topic_id)], topic_prob))
        return topic_embedding_prob_list

    def cal_topic_embedding(self, device):
        all_topic_frequent_words_prob_embeddings = []
        for frequent_words_probabilities, coherence in tqdm(self.dialog_data_topic):
            cur_topic_frequent_words_prob_embeddings = []
            cur_total_probability = 0
            for probability, word in frequent_words_probabilities:
                cur_total_probability += probability
            for probability, word in frequent_words_probabilities:
                word_encoded = self.tokenizer.encode_plus(word, return_tensors="pt", add_special_tokens=False).to(device)
                with torch.no_grad():
                    outputs = self.bert_model(**word_encoded)
                    word_embedding = outputs[0]
                    cur_word_embedding_with_prob = (probability/cur_total_probability * word_embedding.squeeze(0)).mean(dim=0)
                cur_topic_frequent_words_prob_embeddings.append(cur_word_embedding_with_prob)
            all_topic_frequent_words_prob_embeddings.append(torch.mean(torch.stack(cur_topic_frequent_words_prob_embeddings), dim=0))
        return all_topic_frequent_words_prob_embeddings

    def convert_topic_embedding_matrix(self, topic_embedding, context_topic_distribution, response_topic_distribution, persona_topic_distribution, device):
        topic_embedding_data = []
        for context, response, persona in zip(context_topic_distribution, response_topic_distribution, persona_topic_distribution):
            context_topic_embedding_prob = self.convert_topic_id_to_embedding(context, topic_embedding)
            response_topic_embedding_prob = self.convert_topic_id_to_embedding(response, topic_embedding)
            persona_topic_embedding_prob = self.convert_topic_id_to_embedding(persona, topic_embedding)
            topic_embedding_data.append((context_topic_embedding_prob, response_topic_embedding_prob, persona_topic_embedding_prob))
        # all_Uce_context = torch.zeros([len(data), 100, 768], dtype=torch.float64)
        # all_UPct_context = torch.zeros([len(data), 100, 768], dtype=torch.float64)
        #
        # all_Uce_response = torch.zeros([len(data), 100, 768], dtype=torch.float64)
        # all_UPct_response = torch.zeros([len(data), 100, 768], dtype=torch.float64)
        #
        # all_Uce_persona = torch.zeros([len(data), 100, 768], dtype=torch.float64)
        # all_UPct_persona = torch.zeros([len(data), 100, 768], dtype=torch.float64)
        all_Uce_context, all_UPct_context, all_context_mask, \
        all_Uce_response, all_UPct_response, all_response_mask, \
        all_Uce_persona, all_UPct_persona, all_persona_mask = [], [], [], [], [], [], [], [], []

        for context, response, persona in topic_embedding_data:
            # context
            Uce_context = torch.zeros([100, 768], dtype=torch.float)
            UPct_context = torch.zeros([1, 100], dtype=torch.float)
            # Uce_context = np.zeros((100, 768))
            # UPct_context = np.zeros((1, 100))
            context_mask = [1] * len(context) + ([0] * (100 - len(context)))
            for idx, values in enumerate(context):
                topic_id, topic_embedding, topic_prob = values[0], values[1], values[2]
                Uce_context[idx,:] = topic_embedding
                UPct_context[0,idx] = torch.as_tensor(topic_prob)

            all_Uce_context.append(Uce_context)
            all_UPct_context.append(UPct_context)
            all_context_mask.append(context_mask)

            # response
            Uce_response = torch.zeros([100, 768], dtype=torch.float)
            UPct_response = torch.zeros([1, 100], dtype=torch.float)
            # Uce_response = np.zeros((100, 768))
            # UPct_response = np.zeros((1, 100))
            response_mask = [1] * len(response) + ([0] * (100 - len(response)))
            for idx, values in enumerate(response):
                topic_id, topic_embedding, topic_prob = values[0], values[1], values[2]
                Uce_response[idx,:] = topic_embedding
                UPct_response[0,idx] = torch.as_tensor(topic_prob)
            all_Uce_response.append(Uce_response)
            all_UPct_response.append(UPct_response)
            all_response_mask.append(response_mask)

            # persona
            Uce_persona = torch.zeros([100, 768], dtype=torch.float)
            UPct_persona = torch.zeros([1, 100], dtype=torch.float)
            # Uce_persona = np.zeros((100, 768))
            # UPct_persona = np.zeros((1, 100))
            persona_mask = [1] * len(persona) + ([0] * (100 - len(persona)))
            for idx, values in enumerate(persona):
                topic_id, topic_embedding, topic_prob = values[0], values[1], values[2]
                Uce_persona[idx,:] = topic_embedding
                UPct_persona[0,idx] = torch.as_tensor(topic_prob)
            all_Uce_persona.append(Uce_persona)
            all_UPct_persona.append(UPct_persona)
            all_persona_mask.append(persona_mask)

        all_Uce_context = torch.stack(all_Uce_context, dim=0).to(device)
        all_UPct_context = torch.stack(all_UPct_context, dim=0).to(device)
        all_Uce_response = torch.stack(all_Uce_response, dim=0).to(device)
        all_UPct_response = torch.stack(all_UPct_response, dim=0).to(device)
        all_Uce_persona = torch.stack(all_Uce_persona, dim=0).to(device)
        all_UPct_persona = torch.stack(all_UPct_persona, dim=0).to(device)

        all_context_mask = torch.tensor(all_context_mask, dtype=torch.float).to(device)
        all_response_mask = torch.tensor(all_response_mask, dtype=torch.float).to(device)
        all_persona_mask = torch.tensor(all_persona_mask, dtype=torch.float).to(device)

        # cprint("all_Uce_context size: ", all_Uce_context.size(), "all_UPct_context size: ", all_UPct_context.size(), "all_context_mask: ", all_context_mask.size(),\
        #        "all_Uce_response size: ", all_Uce_response.size(), "all_UPct_response size: ", all_UPct_response.size(), "all_response_mask: ", all_response_mask.size(),\
        #        "all_Uce_persona size: ", all_Uce_persona.size(), "all_UPct_persona size: ", all_UPct_persona.size(), "all_persona_mask: ", all_persona_mask.size())
        return all_Uce_context, all_UPct_context, all_context_mask, all_Uce_response, all_UPct_response, all_response_mask, all_Uce_persona, all_UPct_persona, all_persona_mask


    def train_epoch(self, data_iter, models, num_personas, optimizers, schedulers, gradient_accumulation_steps, device, fp16, amp, \
                    apply_interaction, matching_method, aggregation_method):
        epoch_loss = []
        ok = 0
        total = 0
        print_every = 1000
        if len(models) == 1:
            if num_personas == 0:
                context_model, response_model = models[0], models[0]
            else:
                context_model, response_model, persona_model = models[0], models[0], models[0]
        if len(models) == 2:
            context_model, response_model = models
        if len(models) == 3:
            context_model, response_model, persona_model = models

        for optimizer in optimizers:
            optimizer.zero_grad()
        reporter = MemReporter()

        for i, batch in enumerate(data_iter):
            batch = tuple(t.to(device) for t in batch)
            # reporter.report()
            if i%1000 == 0:
                topic_embeddding = self.cal_topic_embedding(device)
            batch_context_word_level = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            batch_response_word_level = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
            batch_context_topic_distribution, batch_response_topic_distribution, batch_persona_topic_distribution =  batch[9],  batch[10], batch[11]
            batch_Uce_context, batch_UPct_context, batch_context_topic_mask, batch_Uce_response, batch_UPct_response, \
            batch_response_topic_mask, batch_Uce_persona, batch_UPct_persona, batch_persona_topic_mask \
                = self.convert_topic_embedding_matrix(topic_embeddding, batch_context_topic_distribution, batch_response_topic_distribution, batch_persona_topic_distribution, device)

            has_persona = len(batch) > 10

            if has_persona:
                batch_word_persona = {"input_ids": batch[6], "attention_mask": batch[7], "token_type_ids": batch[8]}
            output_context_word_level = self.bert_model(**batch_context_word_level)
            output_response_word_level = self.bert_model(**batch_response_word_level)


            batch_context_mask = batch[1].float()
            batch_response_mask = batch[4].float()
            batch_context_emb = output_context_word_level[0] # (batch_size, context_len, emb_size) last hidden state
            batch_response_emb = output_response_word_level[0] # (batch_size, sent_len, emb_size) last hidden state
            batch_size, sent_len, emb_size = batch_response_emb.shape

            batch_persona_emb = None
            batch_persona_mask = None
            num_candidates = batch_size # was batch_size, since we only need one positive case and one negative case
            if has_persona:
                # batch_persona_mask = batch[6].ne(0).float()
                batch_persona_mask = batch[7].float()
                output_persona = self.bert_model(**batch_word_persona)
                batch_persona_emb = output_persona[0] # (batch_size, persona_len, emb_size)
                # persona word level embeddings & Topic embedding
                batch_persona_emb = batch_persona_emb.repeat_interleave(num_candidates, dim=0) # [ABC] => [A*num_candidates B*num_candidates C*num_candidates]
                batch_persona_mask = batch_persona_mask.repeat_interleave(num_candidates, dim=0) # [ABC] => [A*num_candidates B*num_candidates C*num_candidates]
                batch_Uce_persona = batch_Uce_persona.repeat_interleave(num_candidates, dim=0) # [batch_size * num_candidates, 100, 768] # [ABC] => [A*num_candidates B*num_candidates C*num_candidates]
                batch_UPct_persona = batch_UPct_persona.repeat_interleave(num_candidates, dim=0) # [batch_size * num_candidates, 1, 100] # [ABC] => [A*num_candidates B*num_candidates C*num_candidates]
                batch_persona_topic_mask = batch_persona_topic_mask.repeat_interleave(num_candidates, dim=0)

            batch_context_emb = batch_context_emb.repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len, emb_size) # [ABC] => [A*num_candidates B*num_candidates C*num_candidates]
            batch_context_mask = batch_context_mask.repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len) # [ABC] => [A*num_candidates B*num_candidates C*num_candidates]
            batch_Uce_context = batch_Uce_context.repeat_interleave(num_candidates, dim=0) # [batch_size * num_candidates, 100, 768]
            batch_UPct_context = batch_UPct_context.repeat_interleave(num_candidates, dim=0) # [batch_size * num_candidates, 1, 100]
            batch_context_topic_mask = batch_context_topic_mask.repeat_interleave(num_candidates, dim=0)

            batch_response_emb = batch_response_emb.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, sent_len, emb_size) # (batch_size*num_candidates, sent_len, emb_size) # [ABC] => [ABCABCABC]
            batch_response_mask = batch_response_mask.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, sent_len) # (batch_size*num_candidates, sent_len) # [ABC] => [ABCABCABC]

            batch_Uce_response = batch_Uce_response.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, 100, 768) # [batch_size * num_candidates, 100, 768] # [ABC] => [ABCABCABC]
            batch_UPct_response = batch_UPct_response.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, 1, 100) # [batch_size * num_candidates, 1, 100] # [ABC] => [ABCABCABC]
            batch_response_topic_mask = batch_response_topic_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, 100)

            # compute loss
            # targets = torch.arange(batch_size, dtype=torch.long, device=batch[0].device)
            # targets = torch.arange(batch_size, device=batch[0].device)
            targets = torch.eye(batch_size, device=batch[0].device).reshape(batch_size, num_candidates)
            # cprint("targets: ", targets)
            # reporter.report()
            logits = context_model.module.forward (batch_context_emb, batch_response_emb, batch_persona_emb, \
                           batch_context_mask, batch_response_mask, batch_persona_mask, \
                           batch_Uce_context, batch_UPct_context, batch_Uce_response, \
                           batch_UPct_response, batch_Uce_persona, batch_UPct_persona, \
                           batch_context_topic_mask, batch_response_topic_mask, batch_persona_topic_mask, \
                           batch_size, num_candidates, device)
            logits = logits.reshape(batch_size, num_candidates)
            loss = F.cross_entropy(logits, targets)
            # loss = F.binary_cross_entropy_with_logits(logits, targets)
            num_ok = (torch.arange(batch_size, device=batch[0].device).long() == logits.float().argmax(dim=1)).sum()
            ok += num_ok.item()
            total += batch[0].shape[0]
            # reporter.report()
            del logits
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

            else:
                # cprint("loss.backward()")
                loss.backward()
                # for n, p in context_model.named_parameters():
                #     cprint(n, p.grad)

            if (i+1) % gradient_accumulation_steps == 0:
                for model, optimizer, scheduler in zip(models, optimizers, schedulers):
                    if fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    # cprint("optimizer.step()")
                    optimizer.step()
                    scheduler.step()

                    # clear grads here
                    for optimizer in optimizers:
                        optimizer.zero_grad()
            epoch_loss.append(loss.item())

            if i%print_every == 0:
                cprint("train loss: ", np.mean(epoch_loss[-print_every:]))
                cprint("accuracy: ", ok/total)

        acc = ok/total
        return np.mean(epoch_loss), (acc, 0, 0)

    def evaluate_epoch(self, data_iter, models, num_personas, gradient_accumulation_steps, device, dataset, epoch, \
                       apply_interaction, matching_method, aggregation_method):
        epoch_loss = []
        ok = 0
        total = 0
        recall = []
        MRR = []
        print_every = 1000
        if len(models) == 1:
            if num_personas == 0:
                context_model, response_model = models[0], models[0]
            else:
                context_model, response_model, persona_model = models[0], models[0], models[0]
        if len(models) == 2:
            context_model, response_model = models
        if len(models) == 3:
            context_model, response_model, persona_model = models
        # reporter = MemReporter(context_model)
        topic_embeddding = self.cal_topic_embedding(device)
        for batch_idx, batch in enumerate(data_iter):
            # print('========= Report for ', batch_idx, '-th batch in evaluation step =========')
            # reporter.report()
            batch = tuple(t.to(device) for t in batch)
            batch_response_word_level = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
            batch_context_topic_distribution, batch_response_topic_distribution, batch_persona_topic_distribution =  batch[9],  batch[10], batch[11]
            with torch.no_grad():
                batch_Uce_context, batch_UPct_context, batch_context_topic_mask, batch_Uce_response, batch_UPct_response, \
                batch_response_topic_mask, batch_Uce_persona, batch_UPct_persona, batch_persona_topic_mask \
                    = self.convert_topic_embedding_matrix(topic_embeddding, batch_context_topic_distribution, batch_response_topic_distribution, batch_persona_topic_distribution, device)

            has_persona = len(batch) > 10

            # get context embeddings in chunks due to memory constraint
            batch_size = batch[0].shape[0]
            # cprint("batch_size: ", batch_size)
            chunk_size = 20
            num_chunks = math.ceil(batch_size/chunk_size)

            # batch_x_mask = batch[0].ne(0).float()
            # batch_y_mask = batch[3].ne(0).float()
            batch_context_mask = batch[1].float()
            batch_response_mask = batch[4].float()

            batch_context_emb = []
            batch_context_pooled_emb = []
            with torch.no_grad():
                for i in range(num_chunks):
                    mini_batch_context = {
                        "input_ids": batch[0][i*chunk_size: (i+1)*chunk_size],
                        "attention_mask": batch[1][i*chunk_size: (i+1)*chunk_size],
                        "token_type_ids": batch[2][i*chunk_size: (i+1)*chunk_size]
                        }
                    mini_output_x = self.bert_model(**mini_batch_context)
                    batch_context_emb.append(mini_output_x[0]) # [(chunk_size, seq_len, emb_size), ...]
                    batch_context_pooled_emb.append(mini_output_x[1])
                batch_context_emb = torch.cat(batch_context_emb, dim=0) # (batch_size, seq_len, emb_size)
                # batch_Uce_context, batch_UPct_context, batch_context_topic_mask =  batch[9],  batch[10], batch[11].float()
                batch_context_pooled_emb = torch.cat(batch_context_pooled_emb, dim=0)
                emb_size = batch_context_emb.shape[-1]

            if has_persona:
                # batch_persona_mask = batch[6].ne(0).float()
                batch_persona_mask = batch[7].float()
                batch_persona_emb = []
                batch_persona_pooled_emb = []
                with torch.no_grad():
                    for i in range(num_chunks):
                        mini_batch_persona = {
                            "input_ids": batch[6][i*chunk_size: (i+1)*chunk_size],
                            "attention_mask": batch[7][i*chunk_size: (i+1)*chunk_size],
                            "token_type_ids": batch[8][i*chunk_size: (i+1)*chunk_size]
                            }
                        mini_output_persona = self.bert_model(**mini_batch_persona)

                        # [(chunk_size, emb_size), ...]
                        batch_persona_emb.append(mini_output_persona[0])
                        batch_persona_pooled_emb.append(mini_output_persona[1])

                    batch_persona_emb = torch.cat(batch_persona_emb, dim=0)
                    batch_persona_pooled_emb = torch.cat(batch_persona_pooled_emb, dim=0)

            with torch.no_grad():
                output_response = self.bert_model(**batch_response_word_level)
                batch_response_emb = output_response[0]
            batch_size, sent_len, emb_size = batch_response_emb.shape

            # interaction
            # context-response attention
            num_candidates = batch_size
            # cprint("num_candidates: ", num_candidates)
            with torch.no_grad():
                # evaluate per example
                logits = []
                for i in range(batch_size):
                    context_emb = batch_context_emb[i:i+1].repeat_interleave(num_candidates, dim=0) # (num_candidates, context_len, emb_size)
                    context_mask = batch_context_mask[i:i+1].repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len)
                    Uce_context = batch_Uce_context[i:i+1].repeat_interleave(num_candidates, dim=0)
                    UPct_context = batch_UPct_context[i:i+1].repeat_interleave(num_candidates, dim=0)
                    context_topic_mask = batch_context_topic_mask[i:i+1].repeat_interleave(num_candidates, dim=0)
                    persona_emb, persona_mask = None, None
                    if has_persona:
                        persona_emb = batch_persona_emb[i:i+1].repeat_interleave(num_candidates, dim=0)
                        persona_mask = batch_persona_mask[i:i+1].repeat_interleave(num_candidates, dim=0)
                        Uce_persona = batch_Uce_persona[i:i+1].repeat_interleave(num_candidates, dim=0)
                        UPct_persona = batch_UPct_persona[i:i+1].repeat_interleave(num_candidates, dim=0)
                        persona_topic_mask = batch_persona_topic_mask[i:i+1].repeat_interleave(num_candidates, dim=0)
                    logits_single = context_model (context_emb, batch_response_emb, persona_emb, \
                                          context_mask, batch_response_mask, persona_mask, \
                                          Uce_context, UPct_context, batch_Uce_response, \
                                          batch_UPct_response, Uce_persona, UPct_persona, \
                                          context_topic_mask, batch_response_topic_mask, persona_topic_mask, \
                                          batch_size, num_candidates, device)
                    # cprint("logits_single: ", logits_single.shape)
                    logits.append(logits_single)
                logits = torch.stack(logits, dim=0)
                # cprint("logits:", logits.shape)
                # compute loss
                # targets = torch.arange(batch_size, device=batch[0].device)
                targets = torch.eye(batch_size, device=batch[0].device).reshape(batch_size, num_candidates)
                loss = F.cross_entropy(logits, targets)
                # loss = F.cross_entropy(logits, targets)
                # cprint("valid logits: ", logits.shape)
                # cprint("valid targets: ", targets.shape)
                # cprint("valid loss: ", loss.shape)
                # cprint("valid logits: ", logits)
                # cprint("valid targets: ", targets)
                # cprint("valid loss: ", loss)
            num_ok = (torch.arange(batch_size, device=batch[0].device).long() == logits.float().argmax(dim=1)).sum()

            targets = torch.arange(batch_size, dtype=torch.long, device=batch[0].device)
            valid_recall, valid_MRR = compute_metrics_from_logits(logits, targets)

            ok += num_ok.item()
            total += batch[0].shape[0]

            # compute valid recall
            recall.append(valid_recall)
            MRR.append(valid_MRR)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            epoch_loss.append(loss.item())

            if batch_idx%print_every == 0:
                cprint("valid loss: ", np.mean(epoch_loss[-print_every:]))
                cprint("valid recall: ", np.mean(recall[-print_every:], axis=0))
                cprint("valid MRR: ", np.mean(MRR[-print_every:], axis=0))

        acc = ok/total
        # compute recall for validation dataset
        recall = np.mean(recall, axis=0)
        MRR = np.mean(MRR)
        return np.mean(epoch_loss), (acc, recall, MRR)


    def forward(self, batch_context_emb, batch_response_emb, batch_persona_emb, \
         batch_context_mask, batch_response_mask, batch_persona_mask, \
         batch_Uce_context, batch_UPct_context, batch_Uce_response, \
         batch_UPct_response, batch_Uce_persona, batch_UPct_persona, \
         batch_context_topic_mask, batch_response_topic_mask, batch_persona_topic_mask, \
         batch_size, num_candidates, device):

        # word level: [batch_size, # of words]
        # topic_level mask: [batch_size, # of topic K = 100]


        # Bert Word Embedding
        context_response_word_mask = ~torch.bmm(batch_context_mask.unsqueeze(-1), batch_response_mask.unsqueeze(1)).bool()
        persona_response_word_mask = ~torch.bmm(batch_persona_mask.unsqueeze(-1), batch_response_mask.unsqueeze(1)).bool()

        context_response_similarity_matrix = torch.bmm(batch_context_emb, batch_response_emb.transpose(1,2))
        context_response_similarity_matrix = context_response_similarity_matrix.masked_fill_(context_response_word_mask, 0)

        persona_response_similarity_matrix = torch.bmm(batch_persona_emb, batch_response_emb.transpose(1,2))
        persona_response_similarity_matrix = persona_response_similarity_matrix.masked_fill_(persona_response_word_mask, 0)

         # Attention
        context_attn_mask = torch.bmm(batch_context_mask.unsqueeze(-1), batch_context_topic_mask.unsqueeze(1))  # (batch_size, m, n)
        response_attn_mask = torch.bmm(batch_response_mask.unsqueeze(-1), batch_response_topic_mask.unsqueeze(1)) # (batch_size, m, n)
        persona_attn_mask = torch.bmm(batch_persona_mask.unsqueeze(-1), batch_persona_topic_mask.unsqueeze(1))  # (batch_size, m, n)

        context_attn_output = self.context_transformer(batch_context_emb, batch_Uce_context * (batch_UPct_context.repeat(1, 768, 1).transpose(1, 2)), batch_Uce_context, mask=~context_attn_mask.bool())
        response_attn_output = self.response_transformer(batch_response_emb, batch_Uce_response * (batch_UPct_response.repeat(1, 768, 1).transpose(1, 2)), batch_Uce_response, mask=~response_attn_mask.bool())
        persona_attn_output = self.persona_transformer(batch_persona_emb, batch_Uce_persona * batch_UPct_persona.repeat(1, 768, 1).transpose(1, 2), batch_Uce_persona, mask=~persona_attn_mask.bool())

        context_response_attn_mask = ~torch.bmm(context_attn_mask, response_attn_mask.transpose(1, 2)).bool()
        persona_response_attn_mask = ~torch.bmm(persona_attn_mask, response_attn_mask.transpose(1, 2)).bool()

        context_response_attn_similarity_matrix = torch.bmm(context_attn_output, response_attn_output.transpose(1,2))

        context_response_attn_similarity_matrix = context_response_attn_similarity_matrix.masked_fill_(context_response_attn_mask, 0)

        persona_response_attn_similarity_matrix = torch.bmm(persona_attn_output, response_attn_output.transpose(1,2))
        persona_response_attn_similarity_matrix = persona_response_attn_similarity_matrix.masked_fill_(persona_response_attn_mask, 0)

        # Interaction
        # 0. Mask
        response_context_word_mask = ~torch.bmm(batch_response_mask.unsqueeze(-1), batch_context_mask.unsqueeze(1)).bool()
        response_persona_word_mask = ~torch.bmm(batch_response_mask.unsqueeze(-1), batch_persona_mask.unsqueeze(1)).bool()
        # 1. Only word level
        context_add_response_attn = self.context_transformer(batch_context_emb, batch_response_emb, batch_response_emb, mask=~context_response_word_mask.bool())
        response_add_context_attn = self.response_transformer(batch_response_emb, batch_context_emb, batch_context_emb, mask=~response_context_word_mask.bool())

        persona_add_response_attn = self.persona_transformer(batch_persona_emb, batch_response_emb, batch_response_emb, mask=~persona_response_word_mask.bool())
        response_add_persona_attn = self.response_transformer(batch_response_emb, batch_persona_emb, batch_persona_emb, mask=~response_persona_word_mask.bool())
        # 2. mask
        context_response_interaction_mask = ~torch.bmm(batch_context_mask.unsqueeze(-1), batch_response_mask.unsqueeze(1)).bool()
        persona_response_interaction_mask = ~torch.bmm(batch_persona_mask.unsqueeze(-1), batch_response_mask.unsqueeze(1)).bool()

        # 3. similarity matrix
        context_response_word_level_attn_simialrity_matrix = torch.bmm(context_add_response_attn, response_add_context_attn.transpose(1,2))
        context_response_word_level_attn_simialrity_matrix = context_response_word_level_attn_simialrity_matrix.masked_fill_(context_response_interaction_mask, 0)

        persona_response_word_level_attn_simialrity_matrix = torch.bmm(persona_add_response_attn, response_add_persona_attn.transpose(1,2))
        persona_response_word_level_attn_simialrity_matrix = persona_response_word_level_attn_simialrity_matrix.masked_fill_(persona_response_interaction_mask, 0)

        # Topic Attention Interaction
        context_add_response_attn_topic = self.context_transformer(context_attn_output, response_attn_output, response_attn_output, mask=~context_response_word_mask.bool())
        response_add_context_attn_topic = self.response_transformer(response_attn_output, context_attn_output, context_attn_output, mask=~response_context_word_mask.bool())

        persona_add_response_attn_topic = self.persona_transformer(persona_attn_output, response_attn_output, response_attn_output, mask=~persona_response_word_mask.bool())
        response_add_persona_attn_topic = self.response_transformer(response_attn_output, persona_attn_output, persona_attn_output, mask=~response_persona_word_mask.bool())

        context_response_topic_level_attn_simialrity_matrix = torch.bmm(context_add_response_attn_topic, response_add_context_attn_topic.transpose(1,2))
        context_response_topic_level_attn_simialrity_matrix = context_response_topic_level_attn_simialrity_matrix.masked_fill_(context_response_interaction_mask, 0)

        persona_response_topic_level_attn_simialrity_matrix = torch.bmm(persona_add_response_attn_topic, response_add_persona_attn_topic.transpose(1,2))
        persona_response_topic_level_attn_simialrity_matrix = persona_response_topic_level_attn_simialrity_matrix.masked_fill_(persona_response_interaction_mask, 0)

        return self.cnn_block(context_response_attn_similarity_matrix, context_response_similarity_matrix,\
                                persona_response_attn_similarity_matrix, persona_response_similarity_matrix,\
                              context_response_word_level_attn_simialrity_matrix, persona_response_word_level_attn_simialrity_matrix,\
                              context_response_topic_level_attn_simialrity_matrix, persona_response_topic_level_attn_simialrity_matrix).squeeze()



def cprint(*args):
    text = ""
    for arg in args:
        text += "{0} ".format(arg)
    logging.info(text)

def tokenize_conversations(data, tokenizer, max_sent_len):
    new_data = []
    for conv in tqdm(data):
        new_conv = []
        for i, (speaker, sents) in enumerate(conv):
            # each utterance has been segmented into multiple sentences
            if i==0:
                word_limit = 90
            else:
                word_limit = max_sent_len

            tokenized_sent = []
            for sent in sents:
                tokenized = tokenizer.tokenize(sent)
                if len(tokenized_sent) + len(tokenized) <= word_limit:
                    tokenized_sent.extend(tokenized)
                else:
                    break
            if len(tokenized_sent) == 0:
                tokenized_sent = tokenized[:word_limit]
            new_conv.append((speaker, tokenized_sent))
        new_data.append(new_conv)
    return new_data

def tokenize_personas(data, tokenizer, all_speakers, num_personas):
    # average: each speaker corresponds to a list of tokens, separated by [SEP] between sents
    # memnet: each speaker corresponds to a 2D list of tokens
    new_data = {}
    for k, sents in tqdm(data.items()):
        if k in all_speakers:
            tokenized_words = []
            for sent in sents[:num_personas]:

                tokenized_words.extend(tokenizer.tokenize(" ".join(sent))[:22] + ["[SEP]"])
            if len(tokenized_words) > 1:
                tokenized_words.pop() # remove the last [SEP]
                new_data[k] = tokenized_words
            else:
                new_data[k] = ["."]
    return new_data

def create_context_and_response(data):
    new_data = []
    i = 0
    for conv in tqdm(data):
        context = []
        for s, ts in conv[:-1]:
            # 将所有的Context用[SEP]连接在一起 （excluded response)
            context.extend(ts + ["[SEP]"])
        # 将最后一个[SEP]删掉，因为这是最后一句话了，不需要[SEP]了
        i += 1
        if context == []:
            cprint("word skiped id: ", i)
            # cprint("speaker_word: ", conv[-1][0])
            continue
        context.pop() # pop the last [SEP]
        response = conv[-1][1]
        # cprint("response: ", response)
        if len(context) > 0 and len(response) > 0:
            new_data.append((context, response, conv[-1][0]))
        else:
            cprint("word escaped id: ", i)
    return new_data

def create_context_and_response_topic_modelling(data):
    new_data = []
    i = 0
    for conv in tqdm(data):
        context = []
        for s, ts in conv[:-1]:
            # utterance_list = [' '.join([str(item) for item in ts]).strip('\"')]
            # cprint("utterance_list: ", utterance_list)
            context.extend(ts+["."])
        i += 1
        if context == []:
            cprint("topic skiped id: ", i)
            # cprint("speaker_topic: ", conv[-1][0])
            continue
        context.pop() # pop the last [SEP]
        response = conv[-1][1]
        if len(context) > 0 and len(response) > 0:
            new_data.append((" ".join(context), " ".join(response), conv[-1][0]))
        else:
            cprint("topic escaped id: ", i)
            # cprint("new_data: ", new_data)
    return new_data

def convert_conversations_to_ids(data, persona, tokenizer, max_seq_len, max_sent_len, num_personas):
    def pad_tokens(tokens, max_len, sentence_type, num_personas=0, response_ids=None):
        # note token_type_ids to differentiate context utterances
        # speaker A has 0, speaker B has 1, response is speaker B and has 1, persona has 1
        # persona does not have positional embedding
        if sentence_type == "persona" and num_personas > 0:
            # filter persona sentences that appeared in response_ids
            if response_ids is not None:
                response_sent = " ".join(tokenizer.convert_ids_to_tokens(response_ids, skip_special_tokens=True))

                all_persona_sent_ids = []
                for t_id in tokens:
                    if t_id in [101]:
                        sent_ids = []
                    if t_id in [102]:
                        all_persona_sent_ids.append(sent_ids)
                        sent_ids = []
                    if t_id not in tokenizer.all_special_ids:
                        sent_ids.append(t_id)

                # convert ids to tokens
                filtered_tokens = []
                for sent_ids in all_persona_sent_ids:
                    sent = " ".join(tokenizer.convert_ids_to_tokens(sent_ids))
                    if sent not in response_sent:
                        filtered_tokens.extend(sent_ids + [tokenizer.convert_tokens_to_ids("[SEP]")])
                filtered_tokens.insert(0, tokenizer.convert_tokens_to_ids("[CLS]"))

                tokens = filtered_tokens

            # remove additional persona sentences
            persona_sent_count = 0
            truncated_tokens = []
            for token_id in tokens:
                if token_id == tokenizer.convert_tokens_to_ids("[SEP]"):
                    persona_sent_count += 1
                    if persona_sent_count == num_personas:
                        break
                truncated_tokens.append(token_id)
            tokens = truncated_tokens

        assert max_len >= len(tokens)
        attention_mask = [1]*len(tokens)
        padding_length = max_len - len(tokens)
        attention_mask = attention_mask + ([0] * padding_length)

        if sentence_type == "context":
            token_type_ids = []
            token_type = 0
            for token_id in tokens:
                token_type_ids.append(token_type)
                if token_id == tokenizer.convert_tokens_to_ids("[SEP]"):
                    token_type = int(1-token_type)
            token_type_ids = token_type_ids + [0] * padding_length
        else:
            token_type_ids = [0] * max_len

        tokens = tokens + [0] * padding_length
        return tokens, attention_mask, token_type_ids

    all_context_ids = []
    all_context_attention_mask = []
    all_context_token_type_ids = []
    all_response_ids = []
    all_response_attention_mask = []
    all_response_token_type_ids = []
    all_persona_ids = []
    all_persona_attention_mask = []
    all_persona_token_type_ids = []
    max_persona_len = 23*num_personas+1
    context_lens = []
    for context, response, speaker in tqdm(data):
        context_ids = tokenizer.encode(context, add_special_tokens=True) # convert to token ids, add [cls] and [sep] at beginning and end
        response_ids = tokenizer.encode(response, add_special_tokens=True)
        context_lens.append(len(context_ids))

        context_ids, context_attention_mask, context_token_type_ids = pad_tokens(context_ids, max_seq_len, "context")
        response_ids, response_attention_mask, response_token_type_ids = pad_tokens(response_ids, max_sent_len+2, "response")

        all_context_ids.append(context_ids)
        all_context_attention_mask.append(context_attention_mask)
        all_context_token_type_ids.append(context_token_type_ids)
        all_response_ids.append(response_ids)
        all_response_attention_mask.append(response_attention_mask)
        all_response_token_type_ids.append(response_token_type_ids)

        if persona is not None:
            persona_ids = tokenizer.encode(persona[speaker], add_special_tokens=True)
            persona_ids, persona_attention_mask, persona_token_type_ids = pad_tokens(persona_ids, max_persona_len, "persona", num_personas, response_ids)
            # persona_ids, persona_attention_mask, persona_token_type_ids = pad_tokens(persona_ids, max_persona_len, "persona", num_personas)
            all_persona_ids.append(persona_ids)
            all_persona_attention_mask.append(persona_attention_mask)
            all_persona_token_type_ids.append(persona_token_type_ids)

    # (num_examples, max_seq_len)
    all_context_ids = torch.tensor(all_context_ids, dtype=torch.long)
    all_context_attention_mask = torch.tensor(all_context_attention_mask, dtype=torch.long)
    all_context_token_type_ids = torch.tensor(all_context_token_type_ids, dtype=torch.long)

    # (num_examples, max_sent_len)
    all_response_ids = torch.tensor(all_response_ids, dtype=torch.long)
    all_response_attention_mask = torch.tensor(all_response_attention_mask, dtype=torch.long)
    all_response_token_type_ids = torch.tensor(all_response_token_type_ids, dtype=torch.long)

    if persona is not None:
        # (num_examples, max_persona_len)
        all_persona_ids = torch.tensor(all_persona_ids, dtype=torch.long)
        all_persona_attention_mask = torch.tensor(all_persona_attention_mask, dtype=torch.long)
        all_persona_token_type_ids = torch.tensor(all_persona_token_type_ids, dtype=torch.long)

    cprint("all_context_ids: ", all_context_ids.shape, "all_context_attention_mask: ", all_context_attention_mask.shape, "all_context_attention_mask: ", all_context_token_type_ids.shape)
    cprint("all_response_ids: ", all_response_ids.shape, "all_response_attention_mask: ", all_response_attention_mask.shape, "all_response_token_type_ids: ", all_response_token_type_ids.shape)
    cprint("data[99]: ", data[99])
    cprint("all_context_ids: ", all_context_ids[99])
    cprint("all_response_ids: ", all_response_ids[99])
    if persona is not None:
        cprint(all_persona_ids.shape, all_persona_attention_mask.shape, all_persona_token_type_ids.shape)
        # dataset = TensorDataset(all_context_ids, all_context_attention_mask, all_context_token_type_ids, \
        #     all_response_ids, all_response_attention_mask, all_response_token_type_ids, \
        #         all_persona_ids, all_persona_attention_mask, all_persona_token_type_ids)
        return all_context_ids, all_context_attention_mask, all_context_token_type_ids, \
               all_response_ids, all_response_attention_mask, all_response_token_type_ids, \
               all_persona_ids, all_persona_attention_mask, all_persona_token_type_ids
    else:
        # dataset = TensorDataset(all_context_ids, all_context_attention_mask, all_context_token_type_ids, \
        #     all_response_ids, all_response_attention_mask, all_response_token_type_ids)
        return all_context_ids, all_context_attention_mask, all_context_token_type_ids, \
               all_response_ids, all_response_attention_mask, all_response_token_type_ids

def lda_preprocess(docs, common_dic):
    tokenizer = RegexpTokenizer(r'\w+')

    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    docs = [[token for token in doc if not token in stop_words] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    if common_dic == None:
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        cprint('Number of unique tokens: %d' % len(dictionary))
        cprint('Number of documents: %d' % len(corpus))
        return corpus, dictionary
    else:
        corpus = [common_dic.doc2bow(doc) for doc in docs]
        return corpus, None


def check_empty_topic_distribution(topic_distrbution):
    empty_topic_distribution_count = 0
    for tensor in topic_distrbution:
        if len(tensor) == 0:
            empty_topic_distribution_count += 1
    return empty_topic_distribution_count

def merge_persona_token_to_sents (persona, all_speakers, num_personas):
    new_persona={}
    for k, sents in tqdm(persona.items()):
        if k in all_speakers:
            all_sentes = []
            for sent in sents[:num_personas]:
                all_sentes.append(" ".join(sent))
            if len(all_sentes) > 1:
                all_sentes.pop() # remove the last [SEP]
                new_persona[k] = all_sentes
            else:
                new_persona[k] = ["."]
    return new_persona

def generate_data_topic_distribuition(data, persona, lda, common_dict):
    context_topic_distribution, response_topic_distribution, persona_topic_distribution = [], [], []
    for context, response, speaker in data:
        # print("context: ", context)
        # print("response: ", response)
        # print("speaker: ", speaker)
        persoa_sentences = " ".join(persona[speaker])
        context_corpu, context_dic  = lda_preprocess([context], common_dict)
        response_corpus, response_dic = lda_preprocess([response], common_dict)
        persona_corpus, persona_dic = lda_preprocess([persoa_sentences], common_dict)
        context_topic_distribution.append(torch.as_tensor(lda.get_document_topics(context_corpu[0])))
        response_topic_distribution.append(torch.as_tensor(lda.get_document_topics(response_corpus[0])))
        persona_topic_distribution.append(torch.as_tensor(lda.get_document_topics(persona_corpus[0])))
        # cprint(lda.get_document_topics(response_corpus[0]))

    context_empty_distribution_num = check_empty_topic_distribution(context_topic_distribution)
    response_empty_distribution_num = check_empty_topic_distribution(response_topic_distribution)
    persona_empty_distribution_num = check_empty_topic_distribution(persona_topic_distribution)
    cprint("The number of empty topic distribution in context: ", context_empty_distribution_num)
    cprint("The number of empty topic distribution in response: ", response_empty_distribution_num)
    cprint("The number of empty topic distribution in persona: ", persona_empty_distribution_num)

    context_topic_distribution = torch.nn.utils.rnn.pad_sequence(context_topic_distribution, batch_first=True, padding_value=0)
    response_topic_distribution = torch.nn.utils.rnn.pad_sequence(response_topic_distribution, batch_first=True, padding_value=0)
    persona_topic_distribution = torch.nn.utils.rnn.pad_sequence(persona_topic_distribution, batch_first=True, padding_value=0)
    cprint("context_topic_distribution: ", context_topic_distribution.type())
    cprint("response_topic_distribution: ", response_topic_distribution.type())
    cprint("persona_topic_distribution: ", persona_topic_distribution.type())
    return context_topic_distribution, response_topic_distribution, persona_topic_distribution

