# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 6:37 PM
# @Author  : liuxiyang

from asyncio.log import logger
from turtle import color
from helper import *
from data_loader import *

# sys.path.append('./')
from model.models import *
import traceback
import torch.optim as optim
from sklearn import metrics


class Runner(object):

    def load_data(self,trial):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits

        """
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            # for line in open('./data/extended_triples/{}_ext.txt'.format(split)):
            for line in open('./data/ehr/{}_new.txt'.format(split)):
                sub, rel, obj,freq = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        # embed_dim is the Embedding dimension to give as input to score function
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        # jsonFilePath1 = r'data/conversion_dicts/ent_to_id_ext.json'
        # jsonFilePath2 = r'data/conversion_dicts/id_to_ent_ext.json'

        # with open(jsonFilePath1, 'w', encoding='utf-8') as jsonf:
        #     jsonf.write(json.dumps(self.ent2id, indent=2))
        
        # with open(jsonFilePath2, 'w', encoding='utf-8') as jsonf:
        #     jsonf.write(json.dumps(self.id2ent, indent=2))
        # print("Done creating .json files")

        
        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
        # split='train'
            for line in open('./data/ehr/{}_new.txt'.format(split)):
        # for line in open('./data/ehr/exp_train.txt'):
                sub, rel, obj,freq = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)
        # self.data: all origin train + valid + test triplets
        self.data = dict(self.data)
        # # self.sr2o: train origin edges and reverse edges
        self.sr2o = {k: list(v) for k, v in sr2o.items()} #each unique (sub,rel) has list of obj
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        # for (sub, rel), obj in self.sr2o.items():
        #     self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        if self.p.strategy == 'one_to_n':
            for (sub, rel), obj in self.sr2o.items():
                # here label is object list
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        else:
            for sub, rel, obj in self.data['train']:
                rel_inv = rel + self.p.num_rel
                sub_samp = len(self.sr2o[(sub, rel)]) + len(self.sr2o[(obj, rel_inv)])
                sub_samp = np.sqrt(1 / sub_samp)
                
                #TODO: debug below block
                self.triples['train'].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': sub_samp})
                self.triples['train'].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o[(obj, rel_inv)], 'sub_samp': sub_samp})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            # if split=='train':
            #     batch_size = trial.suggest_int("batch_size", 56, 1024)
            #     # batch_size = trial.suggest_int("batch_size", 56, 2048)
            # self.logger.debug(f"Trying with {batch_size} batch size")

            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.test_batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.test_batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.test_batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.test_batch_size),
        }

        self.edge_index, self.edge_type,self.nx_g = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []
        g = nx.Graph()

        # labels = {}
        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)
            # labels[sub] = self.id2ent[sub]
            # labels[obj] = self.id2ent[obj]
            # g.add_node(sub)
            # g.add_node(obj)
            # g.add_edge(sub,obj,edge_type=self.id2rel[rel])
            # net.add_node(sub,label=(self.id2ent[sub]).upper())
            # net.add_node(obj,label=(self.id2ent[obj]).upper())
            # net.add_edge(sub,obj)

        # net.show('edges.html')
        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)
        # edge_index: 2 * 2E, edge_type: 2E * 1, converts list of indices of [(sub1,obj1),(sub2,onj2)]
        edge_index = torch.LongTensor(edge_index).to(self.device).t() # tensor of [[sub1,sub2],[obj1,obj2]]
        edge_type = torch.LongTensor(edge_type).to(self.device) #which is the COO format

        return edge_index, edge_type, g

# for optuna:
    def objective(self,trial):
        # self.p.lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        # self.p.batch_size = trial.suggest_int("batch_size", 56, 1024)
        # # self.p.head_num = trial.suggest_int("head_num", 1, 3)
        # self.p.lbl_smooth = trial.suggest_loguniform("lbl_smooth",4e-2,4e-1)
        # self.p.l2 = trial.suggest_loguniform("l2", 1e-20, 1e-1)
        # self.p.dropout = trial.suggest_uniform("dropout",0.0,0.5)
        # self.p.hid_drop = trial.suggest_uniform("hid_drop",0.0,0.5)

        print(f"Trial params: lr={self.p.lr},bs = {self.p.batch_size}, drop={self.p.dropout}\
        ls={self.p.lbl_smooth},l2={self.p.l2}, hid_drop={self.p.hid_drop}")
        # self.p.init_dim = trial.suggest_int("init_dim", 50, 200)
        # self.p.embed_dim = trial.suggest_int("embed_dim", 100, 300)

        
        self.load_data(trial)
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.optimizer = self.add_optimizer(self.model.parameters(),trial)
        # model = self.model
        try:
            self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
            save_path = os.path.join('./checkpoints', self.p.name)

            if self.p.restore:
                print(f"Save path is = {save_path}")
                self.load_model(save_path)
                self.logger.info('Successfully Loaded previous model')
            val_results = {}
            val_results['mrr'] = 0
            last_val_loss = 100 #for early stopping
            patience = 8
            trigger_times = 0
            self.logger.info(f"Beginning of trial {trial.number}")
            for epoch in range(self.p.max_epochs):
                train_loss = self.run_epoch(epoch, val_mrr) #training
                val_results,val_loss = self.evaluate('valid', epoch) #validation
                print('The Current validation Loss:', val_loss)

                if val_results['mrr'] > self.best_val_mrr:
                    self.best_val = val_results
                    self.best_val_mrr = val_results['mrr']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                
                # Report the best mrr on validation dataset
                trial.report(round(self.best_val_mrr,4),epoch)
                # if trial.should_prune():
                #     self.logger.warning(f"Trial {trial.number} is going to be pruned")
                #     raise optuna.exceptions.TrialPruned()

                self.logger.info(
                    '[Epoch {}]: Training Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss,
                                                                                           self.best_val_mrr))
                if val_loss > last_val_loss: #early stopping
                    trigger_times += 1
                    print('Trigger Times:', trigger_times)
                    if trigger_times >= patience:
                        print('Early stopping!\nStart to test process.')
                        break
                # if val_results['mrr'] < self.best_val_mrr: #early stopping based on mrr
                #     trigger_times += 1
                #     print('Trigger Times:', trigger_times)
                #     if trigger_times >= patience:
                #         print('Early stopping!\nStart to test process.')
                #         break
                else:
                    print('trigger times: 0')
                    trigger_times = 0

                last_val_loss = val_loss
            # self.explain_ragat()
            self.logger.info('Loading best model, Evaluating on Test data')
            self.load_model(save_path)
            test_results,_ = self.evaluate('test', self.best_epoch) #testing
            return self.best_val_mrr
        except Exception as e:
            self.logger.debug("%s____%s\n"
                              "traceback.format_exc():____%s" % (Exception, e, traceback.format_exc()))

    # def explain_ragat(self):
    #     explainer = GNNExplainer(self.model, epochs=self.p.max_epochs, return_type='prob')
    #     node_idx = 1037
    #     train_iter = iter(self.data_iter['train'])

    #     for step, batch in enumerate(train_iter):
    #         self.optimizer.zero_grad()
    #         sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')
    #         node_feat_mask, edge_mask = explainer.explain_node(node_idx, sub, self.edge_index)
    #         ax, G = explainer.visualize_subgraph(node_idx, self.edge_index, edge_mask, y=label)
    #         plt.savefig('plots/rgat_explained.png')
    #     # model.fit(trial)

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        study = optuna.create_study(direction="maximize")
        # uncomment for HPO using Optuna
        # study = optuna.create_study(storage="sqlite:///ragat_trials.db",
        #             study_name="hpos_lblldh_p_rev",direction="maximize",load_if_exists=True,
        #             pruner=optuna.pruners.MedianPruner(n_startup_trials=100))
        # study = optuna.load_study(study_name="hpos_lblldh", storage="sqlite:///ragat_trials.db")
        study.optimize(self.objective, n_trials=1)
        # assert len(loaded_study.trials) == len(study.trials)

        # self.plot_hpo_study(study)
        

    def add_model(self, model, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """
        model_name = '{}_{}'.format(model, score_func)

        if model_name.lower() == 'ragat_transe':
            model = RagatTransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'ragat_distmult':
            model = RagatDistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'ragat_conve':
            model = RagatConvE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'ragat_interacte':
            model = RagatInteractE(self.edge_index, self.edge_type, params=self.p,nx_g=self.nx_g)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, parameters,trial):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        # self.logger.debug(f"Trying with {self.p.lr} learning rate")
        # return getattr(optim, optimizer_name)(parameters, lr=self.p.lr, weight_decay=self.p.l2)
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        Head, Relation, Tails, labels
        """
        # if split == 'train':
        #     triple, label = [_.to(self.device) for _ in batch]
        #     return triple[:, 0], triple[:, 1], triple[:, 2], label
        # else:
        #     triple, label = [_.to(self.device) for _ in batch]
        #     return triple[:, 0], triple[:, 1], triple[:, 2], label
        if split == 'train':
            if self.p.strategy == 'one_to_x':
                triple, label, neg_ent, sub_samp = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, neg_ent, sub_samp
            else:
                triple, label = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        left_results,loss_l = self.predict(split=split, mode='tail_batch')
        right_results,loss_r = self.predict(split=split, mode='head_batch')
        if split == 'test':
            self.classification_metrics(left_results)
            # write n best predictions for each sub,rel to a file
            # self.write_best_preds(left_results,10)
        results = get_combined_results(left_results, right_results)
        
        res_mrr = '\n\tMRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mrr'],
                                                                              results['right_mrr'],
                                                                              results['mrr'])
        res_mr = '\tMR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mr'],
                                                                          results['right_mr'],
                                                                          results['mr'])
        res_hit1 = '\tHit-1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@1'],
                                                                               results['right_hits@1'],
                                                                               results['hits@1'])
        res_hit3 = '\tHit-3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@3'],
                                                                               results['right_hits@3'],
                                                                               results['hits@3'])
        res_hit10 = '\tHit-10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(results['left_hits@10'],
                                                                               results['right_hits@10'],
                                                                               results['hits@10'])
        rel1 = '\n\tMR and MRR for {} : {}, {}\n'.format(self.id2rel[0], results['mr_r1'], results['mrr_r1'])
        rel2 = '\tMR and MRR for {} : {}, {}\n'.format(self.id2rel[1], results['mr_r2'], results['mrr_r2'])
        rel3 = '\tMR and MRR for {} : {}, {}\n'.format(self.id2rel[2], results['mr_r3'], results['mrr_r3'])
        rel4 = '\tMR and MRR for {} : {}, {}\n'.format(self.id2rel[3], results['mr_r4'], results['mrr_r4'])
        # if self.id2rel[4]:
        # rel5 = '\tMR and MRR for {} : {}, {}\n'.format(self.id2rel[4], results['mr_r5'], results['mrr_r5'])
        # rel6 = '\tMR and MRR for {} : {}, {}\n'.format(self.id2rel[5], results['mr_r6'], results['mrr_r6'])

        # rel1_hits = '\tHit-1,Hit-3 and Hit-10 for {} : {}, {}, {}\n'.format(self.id2rel[0], 
        #                             results['hits@1_r1'], results['hits@3_r1'],results['hits@10_r1'])
        # rel2_hits = '\tHit-1,Hit-3 and Hit-10 for {} : {}, {}, {}\n'.format(self.id2rel[1], 
        #                             results['hits@1_r2'], results['hits@3_r2'],results['hits@10_r2'])
        # rel3_hits = '\tHit-1,Hit-3 and Hit-10 for {} : {}, {}, {}\n'.format(self.id2rel[2], 
        #                             results['hits@1_r3'], results['hits@3_r3'],results['hits@10_r3'])
        # rel4_hits = '\tHit-1,Hit-3 and Hit-10 for {} : {}, {}, {}\n'.format(self.id2rel[3], 
        #                             results['hits@1_r4'], results['hits@3_r4'],results['hits@10_r4'])                    
        # rel5_hits = '\tHit-1,Hit-3 and Hit-10 for {} : {}, {}, {}\n'.format(self.id2rel[4], 
        #                             results['hits@1_r5'], results['hits@3_r5'],results['hits@10_r5'])
        # rel6_hits = '\tHit-1,Hit-3 and Hit-10 for {} : {}, {}, {}\n'.format(self.id2rel[5], 
        #                             results['hits@1_r6'], results['hits@3_r6'],results['hits@10_r6'])

        log_res = res_mrr + res_mr + res_hit1 + res_hit3 + res_hit10
        # log_rel_wise = rel1 + rel2 + rel3 + rel4 + rel5 + rel6 + \
        #                     rel1_hits + rel2_hits + rel3_hits + rel4_hits + rel5_hits + rel6_hits
        log_rel_wise = rel1 + rel2 + rel3 + rel4 
                                # + rel5 + rel6
        if (epoch + 1) % 10 == 0 or split == 'test':
            self.logger.info(
                '[Evaluating Epoch {} {}]: {}'.format(epoch, split, log_res))
            self.logger.info("Relation wise evaluation: {}".format(log_rel_wise))
        else:
            self.logger.info(
                '[Evaluating Epoch {} {}]: {}'.format(epoch, split, res_mrr))

        return results, (loss_l + loss_r)/2

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()
        loss_total = 0

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            empty_tensor = torch.tensor([],device=self.device)

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred = self.model.forward(sub, rel)
                if split == 'valid': #early stopping
                    val_loss = self.model.loss(pred, label)
                    loss_total += val_loss.item()
                    # print(f"Here val loss ={loss_total}")
                    # print(f"Len of val_iter = {len(train_iter)}")
                b_range = torch.arange(pred.size()[0], device=self.device)
                if split == 'test' and mode == 'tail_batch':
                # if mode == 'tail_batch':
                    results['all_labels'] = torch.cat((label,results.get('all_labels',empty_tensor)))
                    results['all_preds'] = torch.cat((pred,results.get('all_preds',empty_tensor)))
                    results['all_rels'] = torch.cat((rel, results.get('all_rels',empty_tensor)))
                    results['all_subs'] = torch.cat((sub,results.get('all_subs', empty_tensor)))
                    # if self.id2rel
                    # results['top_pred'] = torch.flatten(torch.cat(torch.argsort(
                    #         pred, dim=1, descending=True)[b_range,0], results.get('top_pred',empty_tensor)))
                    # self.classification_metrics(label, pred,step,all_labels)
                target_pred = pred[b_range, obj]
                # filter setting
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                ranks = ranks.float()
                # print(f"Total no of ranking candidates = {label}")
                # print(f"Predicted score of target object = {pred[b_range,obj].mean()}")
                
                # ranked_better = pred[pred>target_pred.unsqueeze(1).expand(-1,pred.size()[1])]
                # print(f"Rankings = {ranks}")

                if mode == 'tail_batch':
                    results['ranks'] = torch.cat((ranks,results.get('ranks',empty_tensor)))
                    self.rel_specific(results, rel, ranks)
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

                # if step % 100 == 0:
                #     self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        return results, loss_total/len(train_iter)

    def classification_metrics(self,results):
        labels = torch.flatten(results['all_labels']).cpu().data.numpy()
        preds = torch.flatten(results['all_preds']).cpu().data.numpy()
        # self.metrics_for_one_triple(results)

        # 
        # self.metrics_at_k(results,10,-1)
        # self.class_metrics_rel(results,0)
        # self.class_metrics_rel(results,1)
        # self.class_metrics_rel(results,2)
        # self.class_metrics_rel(results,3)
        # self.class_metrics_rel(results,4)
        # self.class_metrics_rel(results,5)

        # 
        # fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        # # roc_auc = metrics.auc(fpr, tpr)
        # roc_auc = metrics.roc_auc_score(labels, preds)
        # precision,recall,thresholds_pr = metrics.precision_recall_curve(labels, preds)
        # # sorted_idx = np.argsort(precision)
        # auc_pr = metrics.auc(recall,precision)
        # # keeping a threshold of 0.5 to predict a link
        # pred_labels = np.where(preds<=0.50,preds*0,preds/preds)
        # f1_score = metrics.f1_score(labels,pred_labels)
        # confusion_matrix = metrics.confusion_matrix(labels,pred_labels)
        # print(f"FPR is ={fpr}")
        # print(f"TPR is = {tpr}")
        # print(f"Thresholds are = {thresholds}")
        fpr_r,tpr_r,precision_r,recall_r, roc_r = np.load('vars_to_plot_ragat.npy',allow_pickle=True)
        auc_pr_r = metrics.auc(recall_r, precision_r)
        print(f"Roc_auc for RAGAT = {roc_r}")
        print(f"AUC_PR for RAGAT = {auc_pr_r}")
        # print(f"F1 score is = {f1_score}")
        # print(f"Confusion matrix:\n {confusion_matrix}")
        # np.save('vars_to_plot_ragat.npy', [fpr, tpr, precision, recall, roc_auc])
        fpr_ext,tpr_ext,precision_ext,recall_ext, roc_ext = np.load('vars_to_plot_ext.npy',allow_pickle=True)
        auc_pr_ext = metrics.auc(recall_ext, precision_ext)
        
        # # for TransE
        # fpr_t,tpr_t,precision_t,recall_t = np.load('vars_for_plot.npy',allow_pickle=True)
        # roc_t, no_skill_t = np.load('vars2.npy',allow_pickle=True)
        # auc_pr_t = metrics.auc(recall_t,precision_t)
        # # for DistMult
        # fpr_d,tpr_d,precision_d,recall_d, roc_d = np.load('vars_to_plot_DistMult.npy',allow_pickle=True)
        # auc_pr_d = metrics.auc(recall_d, precision_d)
        # # for ComplEx
        # fpr_c,tpr_c,precision_c,recall_c, roc_c = np.load('vars_to_plot_ComplEx.npy',allow_pickle=True)
        # auc_pr_c = metrics.auc(recall_c, precision_c)
        # # To plot
        figure, axes = plt.subplots(1, 2)
        plt1 = axes[0]
        plt2 = axes[1]
        plt1.set_title('Receiver Operating Characteristic')
        plt1.plot(fpr_r, tpr_r  , color='wheat', label = 'AUC original KG= %0.4f' % roc_r)
        plt1.plot(fpr_ext, tpr_ext , color='lavender' ,label = 'AUC Extended KG= %0.4f' % roc_ext)
        # plt1.plot(fpr_d, tpr_d , color='thistle' ,label = 'AUC with DistMult= %0.4f' % roc_d)
        # plt1.plot(fpr_c, tpr_c , color='violet' ,label = 'AUC with ComplEx= %0.4f' % roc_c)
        plt1.legend(loc='best', prop={'size': 8})
        plt1.plot([0, 1], [0, 1],'r--')
        plt1.set_xlim(0, 1)
        plt1.set_ylim(0, 1)
        plt1.set_ylabel('True Positive Rate')
        plt1.set_xlabel('False Positive Rate')
        no_skill = len(labels[labels==1]) / len(labels)

        plt2.set_title('Precision Recall curve')
        plt2.plot([0, 1], [no_skill, no_skill], linestyle='--', color='red' ,label='No Skill',linewidth=1)
        plt2.plot(recall_r, precision_r, marker='.',color='lightblue' ,label=f'AUCPR original KG={round(auc_pr_r,4)}')
        plt2.plot(recall_ext,precision_ext,marker='.', color='lightgreen' , label=f'AUCPR extended KG={round(auc_pr_ext,4)}')
        # plt2.plot(recall_d,precision_d,marker='.', color='lightgrey' , label=f'AUCPR with DistMult={round(auc_pr_d,4)}')
        # plt2.plot(recall_c,precision_c,marker='.', color='gold' , label=f'AUCPR with ComplEx={round(auc_pr_c,4)}')
        # axis labels
        plt2.set_xlabel('Recall')
        plt2.set_ylabel('Precision')
        # show the legend
        plt2.legend(loc='center left',prop={'size': 8})
        # show the plot
        figure.tight_layout()
        plt.savefig(f"plots/auroc_aucpr_ragats.png")

    def write_best_preds(self,left_results,n):
        all_subs = left_results['all_subs']
        all_rels = left_results['all_rels']
        all_preds = left_results['all_preds']
        all_labels = left_results['all_labels']
        b_range2 = torch.arange(all_preds.size()[0], device=self.device)
        top_pred_idx = torch.argsort(all_preds, dim=1, descending=True)[b_range2,:n]
        pred_triples = []
        for i in range(all_preds.size()[0]):
            t = top_pred_idx[i]
            l = all_labels[i]
            for j in range(top_pred_idx.size()[1]):
                # print(f"one of the pred = {type(all_subs[i])}")
                # if not true label only then add to the predictions
                if l[t[j]] != 1:
                    subject = str(self.id2ent[all_subs[i].item()])
                    relation = str(self.id2rel[all_rels[i].item()])
                    obj = str(self.id2ent[t[j].item()])
                    score = all_preds[i,t[j]].item()
                    pred_triples.append([subject,relation,obj,score])
        df = pd.DataFrame(pred_triples)
        df.to_csv(r'predicted_triples_rev.csv',header=['sub','rel','obj','score'],index=False)


    def rel_specific(self, results, rel, ranks):
        # print(f"All relations = {type(rel)}")
        # print(f"All ranks = {ranks}")
        # below are the indexes of each original relation type
        rel_one = (rel==0).nonzero(as_tuple=True)[0]
        rel_two = (rel==1).nonzero(as_tuple=True)[0]
        rel_three = (rel==2).nonzero(as_tuple=True)[0]
        rel_four = (rel==3).nonzero(as_tuple=True)[0]

        # print(f"Relations of type {self.id2rel[2]} have indexes={rel_three}")
        # print(f"Relations of type {self.id2rel[3]} have indexes={rel_four}")
        ranks_r1 = ranks[rel_one]
        ranks_r2 = ranks[rel_two]
        ranks_r3 = ranks[rel_three]
        ranks_r4 = ranks[rel_four]
        # print(f"Ranks of type {self.id2rel[0]} have ranks={ranks_r1}")
        # print(f"Ranks of type {self.id2rel[3]} have ranks={ranks_r4}")
        results['count_r1'] = torch.numel(ranks_r1) + results.get('count_r1',0.0)
        results['count_r2'] = torch.numel(ranks_r2) + results.get('count_r2',0.0)
        results['count_r3'] = torch.numel(ranks_r3) + results.get('count_r3',0.0)
        results['count_r4'] = torch.numel(ranks_r4) + results.get('count_r4',0.0)

        results['mr_r1'] = torch.sum(ranks_r1).item() + results.get('mr_r1', 0.0)
        results['mr_r2'] = torch.sum(ranks_r2).item() + results.get('mr_r2', 0.0)
        results['mr_r3'] = torch.sum(ranks_r3).item() + results.get('mr_r3', 0.0)
        results['mr_r4'] = torch.sum(ranks_r4).item() + results.get('mr_r4', 0.0)

        results['mrr_r1'] = torch.sum(1.0 / ranks_r1).item() + results.get('mrr_r1', 0.0)
        results['mrr_r2'] = torch.sum(1.0 / ranks_r2).item() + results.get('mrr_r2', 0.0)
        results['mrr_r3'] = torch.sum(1.0 / ranks_r3).item() + results.get('mrr_r3', 0.0)
        results['mrr_r4'] = torch.sum(1.0 / ranks_r4).item() + results.get('mrr_r4', 0.0)

        if 4 in rel:
            rel_five = (rel==4).nonzero(as_tuple=True)[0]
            rel_six = (rel==5).nonzero(as_tuple=True)[0]
            ranks_r5 = ranks[rel_five]
            ranks_r6 = ranks[rel_six]
            results['count_r5'] = torch.numel(ranks_r5) + results.get('count_r5',0.0)
            results['count_r6'] = torch.numel(ranks_r6) + results.get('count_r6',0.0)
            results['mr_r5'] = torch.sum(ranks_r5).item() + results.get('mr_r5', 0.0)
            results['mr_r6'] = torch.sum(ranks_r6).item() + results.get('mr_r6', 0.0)
            results['mrr_r5'] = torch.sum(1.0 / ranks_r5).item() + results.get('mrr_r5', 0.0)
            results['mrr_r6'] = torch.sum(1.0 / ranks_r6).item() + results.get('mrr_r6', 0.0)

        # for k in range(10):
        #     results['hits@{}_r1'.format(k + 1)] = torch.numel(ranks_r1[ranks_r1 <= (k + 1)]) + results.get(
        #         'hits@{}_r1'.format(k + 1), 0.0)
        #     results['hits@{}_r2'.format(k + 1)] = torch.numel(ranks_r2[ranks_r2 <= (k + 1)]) + results.get(
        #         'hits@{}_r2'.format(k + 1), 0.0)       
        #     results['hits@{}_r3'.format(k + 1)] = torch.numel(ranks_r3[ranks_r3<= (k + 1)]) + results.get(
        #         'hits@{}_r3'.format(k + 1), 0.0)
        #     results['hits@{}_r4'.format(k + 1)] = torch.numel(ranks_r4[ranks_r4 <= (k + 1)]) + results.get(
        #         'hits@{}_r4'.format(k + 1), 0.0)
        #     results['hits@{}_r5'.format(k + 1)] = torch.numel(ranks_r5[ranks_r5 <= (k + 1)]) + results.get(
        #         'hits@{}_r5'.format(k + 1), 0.0)
        #     results['hits@{}_r6'.format(k + 1)] = torch.numel(ranks_r6[ranks_r6 <= (k + 1)]) + results.get(
        #         'hits@{}_r6'.format(k + 1), 0.0)    

    def metrics_for_one_triple(self, results):
        label_1 = results['all_labels'][4].cpu().data.numpy()
        pred_1 = results['all_preds'][4].cpu().data.numpy()
        fpr_1,tpr_1,thresholds_1 = metrics.roc_curve(label_1, pred_1)
        roc_auc_1 = metrics.roc_auc_score(label_1, pred_1)
        precision_1,recall_1,thresholds_pr_1 = metrics.precision_recall_curve(label_1, pred_1)
        auc_pr_1 = metrics.auc(recall_1,precision_1)
        print(f"Roc_auc for 1 triple = {roc_auc_1}")
        print(f"Ranks among {len(label_1)} entries")
        print(f"AUC_PR for 1 triple = {auc_pr_1}")
        plt.rcdefaults()
        fig, ax = plt.subplots()
        idxs = np.arange(len(pred_1))
        bars = ax.bar(idxs,pred_1,color=np.where(label_1==1, 'green', 'blue'))
        ax.set_xlabel('objects')
        ax.set_ylabel('scores')
        ax.set_title("Sample scores of all objects given a subject and relation.")
        plt.savefig('plots/sample_scores.png')

    def class_metrics_rel(self, results,relation):
        all_preds = results['all_preds']
        all_labels = results['all_labels']
        all_rels = results['all_rels']
        if relation != -1:
            rel_idx = (all_rels==relation).nonzero(as_tuple=True)[0]
            all_preds = all_preds[rel_idx,:]
            all_labels = all_labels[rel_idx,:]
            # print(f"Predictions of rel {self.id2rel[2]} = {pred_three}")
        # 
        # b_range2 = torch.arange(all_preds.size()[0], device=self.device)
        # top_pred_idx = torch.argsort(all_preds, dim=1, descending=True)[b_range2,:]
        # # print(f"Top pred index = {top_pred_idx}")
        # # top_pred = torch.where(all_preds[b_range,top_pred_idx[b_range]]>0.5, 1, 0)
        # top_pred = torch.zeros(all_preds.size()[0], top_pred_idx.size()[1])
        # top_labels = torch.zeros(all_preds.size()[0], top_pred_idx.size()[1])
        # for i in b_range2:
        #     for idx,j in enumerate(top_pred_idx[i]):
        #         top_pred[i,idx] = all_preds[i,j]
        #         top_labels[i,idx] = all_labels[i,j]
        top_pred = torch.flatten(all_preds).cpu().data.numpy()
        top_labels = torch.flatten(all_labels).cpu().data.numpy()
        fpr_rel, tpr_rel, thresholds = metrics.roc_curve(top_labels, top_pred)
        roc_auc_rel = metrics.auc(fpr_rel, tpr_rel)
        precision_rel,recall_rel,thresholds_pr_rel = metrics.precision_recall_curve(
                                                            top_labels, top_pred)
        auc_pr_rel = metrics.auc(recall_rel, precision_rel)
        print(f"For {self.id2rel[relation]}, AUC_ROC={roc_auc_rel}, AUCPR={auc_pr_rel} ")
        # return roc_auc_rel,auc_pr_rel
                                

    def metrics_at_k(self, results,k,relation=-1):
        all_preds = results['all_preds']
        all_labels = results['all_labels']
        all_rels = results['all_rels']
        # Relation wise classification metrics at k
        if relation != -1:
            rel_idx = (all_rels==relation).nonzero(as_tuple=True)[0]
            all_preds = all_preds[rel_idx,:]
            all_labels = all_labels[rel_idx,:]
            # print(f"Predictions of rel {self.id2rel[2]} = {pred_three}")
        # 
        b_range2 = torch.arange(all_preds.size()[0], device=self.device)
        top_pred_idx = torch.argsort(all_preds, dim=1, descending=True)[b_range2,:k]
        # print(f"Top pred index = {top_pred_idx}")
        # top_pred = torch.where(all_preds[b_range,top_pred_idx[b_range]]>0.5, 1, 0)
        top_pred = torch.zeros(all_preds.size()[0], top_pred_idx.size()[1])
        top_labels = torch.zeros(all_preds.size()[0], top_pred_idx.size()[1])
        for i in b_range2:
            for idx,j in enumerate(top_pred_idx[i]):
                top_pred[i,idx] = all_preds[i,j]
                top_labels[i,idx] = all_labels[i,j]
        top_pred = torch.flatten(top_pred).cpu().data.numpy()
        top_labels = torch.flatten(top_labels).cpu().data.numpy()
        fpr_at_k, tpr_at_k, thresholds = metrics.roc_curve(top_labels, top_pred)
        roc_auc_at_k = metrics.auc(fpr_at_k, tpr_at_k)
        precision_at_k,recall_at_k,thresholds_pr_at_k = metrics.precision_recall_curve(
                                                            top_labels, top_pred)
        top_pred_labels = np.where(top_pred<=0.50,top_pred*0,top_pred/top_pred)
        f1_score_at_k = metrics.f1_score(top_labels,top_pred_labels)
        confusion_matrix_at_k = metrics.confusion_matrix(top_labels,top_pred_labels)
        if relation == -1:
            r = "all relations"
        else:
            r = self.id2rel[relation]
        # print(f"All Predictions = {all_preds}")
        print(f"Roc_auc@{k} for {r} = {roc_auc_at_k}")
        # print(f"Top Predictions = {top_pred}")
        # print(f"All labels = {all_labels}")
        # print(f"Top labels = {top_labels}")
        print(f"F1@{k} for {r} = {f1_score_at_k}")
        print(f"Confusion@{k} for {r} = {confusion_matrix_at_k}")
        figure, axes = plt.subplots(1, 2)
        plt1 = axes[0]
        plt2 = axes[1]
        plt1.set_title(f'Receiver Operating Characteristic')
        plt1.plot(fpr_at_k, tpr_at_k  , 'b', label = 'AUC = %0.2f' % roc_auc_at_k)
        plt1.legend(loc = 'lower right')
        plt1.plot([0, 1], [0, 1],'r--')
        plt1.set_xlim(0, 1)
        plt1.set_ylim(0, 1)
        plt1.set_ylabel('True Positive Rate')
        plt1.set_xlabel('False Positive Rate')
        no_skill = len(top_labels[top_labels==1]) / len(top_labels)
        plt2.set_title(f'Precision Recall curve')
        plt2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill',linewidth=1)
        plt2.plot(recall_at_k, precision_at_k, marker='.', label='RAGAT')
        # axis labels
        plt2.set_xlabel('Recall')
        plt2.set_ylabel('Precision')
        # show the legend
        plt2.legend()
        # show the plot
        plt.savefig(f"plots/auroc_aucpr_at_{k}_{r}.png")


    def run_epoch(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')

            # pred = self.model.forward(sub, rel)
            pred = self.model.forward(sub, rel, neg_ent)
            loss = self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            # if step % 100 == 0:
            #     self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses),
            #                                                                                self.best_val_mrr,
            #                                                                                self.p.name))

        loss = np.mean(losses)
        # self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def plot_hpo_study(self, study):
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        incomplete_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        # trials_df = study.trials_dataframe()
        # trials_df.columns = trials_df.columns.str.replace("test","val")
        # failed_lrs,failed_bs,failed_ls,failed_att,failed_init_dim,failed_out_dim = [],[],[],[],[],[]
        # for trial in incomplete_trials:
        #     failed_lrs.append(trial._user_attrs['lr'])
        #     failed_bs.append(trial._user_attrs['batch_size'])
        # print(f"failed_lrs = {failed_lrs}")
        # print(f"failed_bs = {failed_bs}")

        # trials_df['pruned_lrs'] = failed_lrs
        # trials_df['pruned_bs'] = failed_bs
        # trials_df['best_lr'] = study.best_trial._user_attrs['lr']
        # trials_df['best_batch_size'] = study.best_trial._user_attrs['batch_size']
        # trials_df.to_csv(r'trials/trials_df.csv',sep='\t')
        print("  Number of complete trials: ", len(complete_trials))
        print("Best trial: ")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
        fig = optuna.visualization.plot_optimization_history(study)
        # fig2 = optuna.visualization.plot_contour(study, params=["batch_size", "lr",
        #                                             "lbl_smooth","l2"])
        fig3 = optuna.visualization.plot_slice(study, params=["batch_size", "lr",
                                                    "lbl_smooth","l2","hid_drop","dropout"])
        fig4 = optuna.visualization.plot_param_importances(study)
        fig.write_image("plots/optuna/study_history_lblldh_p_rev.png")
        # fig2.write_image("plots/optuna/study_contour2.png")
        fig3.write_image("plots/optuna/parameter_slice_lblldh_p_rev.png")
        fig4.write_image("plots//optuna/param_imp_lblldh_p_rev.png")
        

    def fit(self,trial):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        # try:
        #     self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        #     save_path = os.path.join('./checkpoints', self.p.name)

        #     if self.p.restore:
        #         self.load_model(save_path)
        #         self.logger.info('Successfully Loaded previous model')
        #     val_results = {}
        #     val_results['mrr'] = 0
        #     for epoch in range(self.p.max_epochs):
        # train_loss = self.run_epoch(epoch, val_mrr)
        # # if ((epoch + 1) % 10 == 0):
        # val_results = self.evaluate('valid', epoch,trial)
        
        # print(f"hi!, the mrr is = {val_results['mrr']}")

        

        # if val_results['mrr'] > self.best_val_mrr:
        #     self.best_val = val_results
        #     self.best_val_mrr = val_results['mrr']
        #     self.best_epoch = epoch
        #     self.save_model(save_path)

        # self.logger.info(
        #     '[Epoch {}]: Training Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss,
        #                                                                             self.best_val_mrr))

        # self.logger.info('Loading best model, Evaluating on Test data')
        # self.load_model(save_path)
        # test_results = self.evaluate('test', self.best_epoch,trial)
        # except Exception as e:
        #     self.logger.debug("%s____%s\n"
        #                       "traceback.format_exc():____%s" % (Exception, e, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model', dest='model', default='ragat', help='Model Name')
    parser.add_argument('-score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='corr', help='Composition Operation to be used in RAGAT')
    # opn is new hyperparameter
    parser.add_argument('-batch', dest='batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('-test_batch', dest='test_batch_size', default=1024, type=int,
                        help='Batch size of valid and test data')
    parser.add_argument('-gamma', type=float, default=40.0, help='Margin')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-init_dim', dest='init_dim', default=100, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', dest='embed_dim', default=200, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.3, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')

    # InteractE hyperparameters
    parser.add_argument('-neg_num', dest="neg_num", default=1000, type=int,
                        help='Number of negative samples to use for loss calculation')
    parser.add_argument("-strategy", type=str, default='one_to_n', help='Training strategy to use')
    parser.add_argument('-form', type=str, default='plain', help='The reshaping form to use')
    parser.add_argument('-ik_w', dest="ik_w", default=10, type=int, help='Width of the reshaped matrix')
    parser.add_argument('-ik_h', dest="ik_h", default=20, type=int, help='Height of the reshaped matrix')
    parser.add_argument('-inum_filt', dest="inum_filt", default=200, type=int, help='Number of filters in convolution')
    parser.add_argument('-iker_sz', dest="iker_sz", default=9, type=int, help='Kernel size to use')
    parser.add_argument('-iperm', dest="iperm", default=1, type=int, help='Number of Feature rearrangement to use')
    parser.add_argument('-iinp_drop', dest="iinp_drop", default=0.3, type=float, help='Dropout for Input layer')
    parser.add_argument('-ifeat_drop', dest="ifeat_drop", default=0.3, type=float, help='Dropout for Feature')
    parser.add_argument('-ihid_drop', dest="ihid_drop", default=0.3, type=float, help='Dropout for Hidden layer')
    parser.add_argument('-attention', dest="att", help="Whether to use attention layer")
    parser.add_argument('-head_num', dest="head_num", default=2, type=int, help="Number of attention heads")

    args = parser.parse_args()

    if not args.restore:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = Runner(args)
    # model.fit()
