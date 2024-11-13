import os
import pickle
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from model import MID
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from torch import nn
from torch.utils import data

gpu_no = 1
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_no}'
os.environ['OMP_NUM_THREADS'] = '8'


test_comp = ['Alectinib', 'Amphotericin B', 'Asciminib', 'Bedaquiline', 'Belzutifan', 'Benzethonium ',
 'chloride', 'Cabozantinib', 'Carvedilol', 'Ceritinib', 'Daclatasvir', 'Dolasetron', 'Domperidone',
 'Droperidol', 'Eltrombopag', 'Erlotinib', 'Erythromycin', 'Fostamatinib', 'Ibrutinib', 'Irbesartan',
 'Itraconazole', 'Lesinurad', 'Miltefosine', 'Mizolastine', 'N-Acetylprocainamide', 'Nintedanib', 'Pimozide',
 'Pitolisant', 'Ponatinib', 'Pretomanid', 'Propranolol', 'Rilpivirine', 'Sildenafil', 'Silodosin', 'Talazoparib', 'Telithromycin',
 'Valbenazine', 'Vesnarinone', 'Vismodegib']


class MIDDataset(data.Dataset):
    def __init__(self,
                 root_path='/home/ubuntu/BX/data/single-cells',
                 meta_path='sc-metadata_old.csv',
                 train_states='train',
                 test_comp=[]):
        self.root_path = root_path

        meta_path = os.path.join(root_path, meta_path)
        self.meta_info = pd.read_csv(meta_path)
        self.img_files = self.meta_info['Image_Name'].values.tolist()
        compound = self.meta_info['Compound_name'].values.tolist()
        target = self.meta_info['Cardioxicity'].values.tolist()

        self.target_map = {y: x for x, y in enumerate(sorted(list(set(target))))}
        self.states = self.meta_info['training'].values.tolist()
        self.cls_num = len(self.target_map)
        if train_states == 'train':
            self.data = [[self.img_files[x], compound[x], self.target_map[target[x]]] for x in range(len(self.img_files)) if
                         compound[x] not in test_comp]
            self.transform = nn.Identity()

        else:
            self.data = [[self.img_files[x], compound[x], self.target_map[target[x]]] for x in range(len(self.img_files)) if
                         compound[x] in test_comp]
            self.transform = nn.Identity()

        self.dequeue = {}
        for i in self.data:
            one_path, one_compound, one_label = i
            if one_compound in self.dequeue.keys():
                self.dequeue[one_compound].append([one_path, one_compound, one_label])
            else:
                self.dequeue[one_compound] = [[one_path, one_compound, one_label]]
        max_len = int(1.5 * np.mean([len(self.dequeue[x]) for x in self.dequeue.keys()]))
        self.data = []
        less_num = []
        for i in self.dequeue:
            if len(self.dequeue[i]) < max_len // 5:
                self.dequeue[i] = self.dequeue[i] * (max_len // len(self.dequeue[i]))
                less_num.append(i)
            self.data.extend([x for x in self.dequeue[i][:max_len]])
        random.shuffle(self.data)
        self.data = self.data[:len(self.data)//2]
        self.p_num = 400
        print('compound num', len(self.dequeue), 'each cls max len', max_len, 'less compound', len(less_num))

    def __getitem__(self, index):
        _, compound, label = self.data[index]
        h, w = 96, 96 * 3
        partner = random.sample(self.dequeue[compound], self.p_num)
        img_names = '\1'.join([x[0] for x in partner])
        partner = [os.path.join(self.root_path, x[0]) for x in partner]
        partner = torch.tensor(np.array([np.asarray(Image.open(x), dtype=np.uint8) for x in partner]))
        partner = partner.reshape(self.p_num, 1, h, 3, w // 3).permute(0, 1, 3, 2, 4).reshape(self.p_num, 3, h, w // 3)
        try:
            partner = self.transform(partner)/255.
        except:
            partner = partner / 255.
        return partner, label, img_names

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    for kk in range(20):
        true_num = 0
        wrong_comp = []
        all_label_lst = []
        all_score_lst = []
        all_res_lst = []
        comp_all_pos = {}
        comp_all_neg = {}
        comp_dict = {'Amphotericin B': '0', 'Astemizole': '1', 'Sertindole': '1', 'Lemborexant': '1', 'DMSO': '0',
                     'Mocetinostat': '0', 'Mepivacaine': '0', 'Betrixaban': '1', 'Benzethonium chloride': '1',
                     'Erlotinib': '0', 'Clotrimazole': '1', 'Erythromycin': '1', 'Propranolol': '1', 'Vinpocetine': '1',
                     'Pitolisant': '1', 'Alectinib': '0', 'Mitoxantrone': '1', 'Itraconazole': '0', 'Sunitinib': '1',
                     'Voriconazole': '0', 'Ceritinib': '0', 'Carvedilol': '1', 'Fluspirilene': '1', 'Lumefantrine': '1',
                     'Dolasetron': '1', 'Cisapride': '1', 'Alfuzosin': '0', 'Lifitegrast': '0', 'Loratadine': '1',
                     'Ondansetron': '1', 'Tamoxifen': '1', 'Talazoparib': '0', 'Lesinurad': '0', 'Nitrendipine': '1',
                     'Omarigliptin': '0', 'Droperidol': '1', 'Dronedarone': '1', 'Gatifloxacin': '0', 'Vorinostat': '0',
                     'Nintedanib': '1', 'Moclobemide': '0', 'Haloperidol': '1', 'Irbesartan': '0', 'Asciminib': '0',
                     'Eltrombopag': '1', 'Bedaquiline': '1', 'Procainamide': '0', 'Silodosin': '0', 'Donepezil': '1',
                     'Epinastine': '0', 'Valbenazine': '1', 'Carbamazepine': '0', 'Quetiapine': '1', 'Domperidone': '1',
                     'Belzutifan': '0', 'N': '0', 'Vismodegib': '0', 'Ibrutinib': '1', 'Mizolastine': '0',
                     'Riluzole': '0', 'Sildenafil': '1', 'Ajmaline': '1', 'Aripiprazole': '1', 'Ranolazine': '0',
                     'Cabozantinib': '0', 'Amsacrine': '1', 'Tadalafil': '0', 'Belinostat': '0',
                     'N-Acetylprocainamide': '0', 'Nomifensine': '0', 'Daclatasvir': '0', 'Ebastine': '1',
                     'Rilpivirine': '1', 'Quinidine': '1', 'Verapamil': '1', 'Linagliptin': '0', 'Terfenadine': '1',
                     'Lopinavir': '1', 'Perphenazine': '1', 'Glibenclamide': '0', 'Disopyramide': '0',
                     'Dofetilide': '1', 'Ziprasidone': '1', 'Lidocaine': '0', 'Miltefosine': '0', 'Ciprofloxacin': '0',
                     'Risperidone': '1', 'Vesnarinone': '1', 'Fostamatinib': '1', 'Tolterodine': '1', 'Darunavir': '0',
                     'Paliperidone': '1', 'Osimertinib': '1', 'Ponatinib': '1', 'Rosiglitazone': '0', 'Pretomanid': '0',
                     'Ketoconazole': '1', 'Solifenacin': '1', 'Pimozide': '1', 'Telithromycin': '0', 'Chloroquine': '1',
                     'Fluvoxamine': '1'}
        b_comp_dict = ['Alectinib', 'Amphotericin B', 'Asciminib', 'Belzutifan',
                       'Benzethonium chloride', 'Cabozantinib', 'Carvedilol', 'Daclatasvir', 'Dolasetron',
                       'Domperidone',
                       'Droperidol', 'Erythromycin', 'Fostamatinib', 'Ibrutinib',
                       'Irbesartan',
                       'Itraconazole', 'Lesinurad', 'Miltefosine', 'Mizolastine', 'N-Acetylprocainamide',
                       'Pimozide',
                       'Pitolisant', 'Ponatinib', 'Pretomanid', 'Propranolol', 'Rilpivirine', 'Sildenafil', 'Silodosin',
                       'Talazoparib', 'Telithromycin',
                       'Valbenazine', 'Vesnarinone', ]

        aaa_lst = []
        for a_comp in b_comp_dict:
            a_test_comp = [a_comp]
            device = 'cuda'
            try:
                f_db = MIDDataset(train_states='test', test_comp=a_test_comp)
            except:
                continue
            # a_label = f_db.compound_real_map[a_comp]
            f_db = data.DataLoader(f_db, batch_size=1, num_workers=16, shuffle=False, prefetch_factor=2)
            f_model = MID(cls_num=2).to('cuda')


            # modelckpt = torch.load('/home/xiangrui.gao/Phenotype/model_zoo/deep_filter_t3_best.pt')
            # modelckpt = torch.load('/home/xiangrui.gao/Phenotype/model_zoo/deep_filter_t3_Card_8.pt')
            # modelckpt = torch.load('/home/xiangrui.gao/Phenotype/model_zoo/deep_filter_t4_Card_
            # best.pt')
            pwd = f'/home/ubuntu/BX/model_zoo/deep_filter_t6_Card_best1.pt'
            modelckpt = torch.load(pwd)
            new_state_dict = OrderedDict()
            for k, v in modelckpt.items():
                name = k[7:]
                new_state_dict[name] = v
            f_model.load_state_dict(new_state_dict, strict=True)
            f_model.eval()
            score_all = []
            img_info = []
            label_lst = []
            res_lst = []
            acc_lst = []
            step = 0
            comp_img_map = []
            cls_lst = []
            sim_emb_lst = []
            nonsim_emb_lst = []
            with torch.no_grad():
                for i in f_db:
                    p_img, label, img_name = i
                    img_name = np.asarray([x.split('\1') for x in img_name])[0]
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            p_img, label = p_img.to(device), label.to(device)
                            embs, attn, outs = f_model(p_img)
                            embs = embs.detach().cpu().numpy()[0]
                            attn = attn[:, :, 0, 1:].detach().cpu().numpy().mean(1)[0]
                            cls_embs = embs[0,]
                            cls_lst.append(cls_embs)
                    max_index = np.argmax(attn)
                    min_index = np.argmin(attn)
                    sim_embs = embs[1:][max_index]
                    nonsim_embs = embs[1:][min_index]
                    sim_emb_lst.append(sim_embs)
                    nonsim_emb_lst.append(nonsim_embs)
                    comp_img_map.extend(zip(img_name, attn))
                    score = outs.softmax(1)[:, 1].reshape(-1).detach().cpu().numpy().tolist()
                    outs = outs.argmax(1).reshape(-1).detach().cpu().numpy().tolist()
                    label = label.reshape(-1).detach().cpu().numpy().tolist()
                    label_lst.extend(label)
                    res_lst.extend(outs)
                    all_score_lst.extend(score)
                    all_label_lst.extend(label)
                    all_res_lst.extend(outs)
                    # print(label, outs)
                    # print(1111, label[:10], outs[:10])
                    acc = accuracy_score(label, outs)
                    acc_lst.append(acc)
                    step += 1
                    if step >50:
                        if np.mean(acc_lst) > 0.5:
                            true_num += 1
                            # print(111, a_comp)
                        else:
                            wrong_comp.append(a_comp)
                            # print(222, a_comp)
                        # print(a_comp, Counter(label_lst), np.mean(acc_lst), Counter(res_lst), true_num, wrong_comp)
                        break
            comp_img_map = sorted(comp_img_map, key=lambda x:x[1], reverse=True)
            comp_all_pos[a_comp] = {'img': comp_img_map[:100], 'emb': cls_lst, 're_emb':sim_emb_lst}
            comp_all_neg[a_comp] = {'img': comp_img_map[-100:], 'emb': cls_lst, 're_emb':nonsim_emb_lst}
            aaa_lst.append([a_comp, np.mean(all_label_lst[-10:]), np.mean(all_score_lst[-50:])])
            print(a_comp, np.mean(all_label_lst[-10:]), np.mean(all_score_lst[-50:]))
        with open('/home/ubuntu/BX/comp_pos_img_single_all.pkl', 'wb') as f:
            pickle.dump(comp_all_pos, f)
        with open('/home/ubuntu/BX/comp_neg_img_single_all.pkl', 'wb') as f:
            pickle.dump(comp_all_neg, f)
        # print(all_label_lst[:20], all_res_lst[:20])
        print(aaa_lst)
        print(1111,
              'recall',
              recall_score(all_label_lst, all_res_lst),
              'prec',
              precision_score(all_label_lst, all_res_lst),
              'auc',
              roc_auc_score(all_label_lst, all_score_lst),
              )
        print(kk, true_num, wrong_comp)
