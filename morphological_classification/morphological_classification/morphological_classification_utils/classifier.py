import sys
import argparse
import pickle
from itertools import product
from copy import deepcopy

import numpy as np
from scipy import stats

# from morphological_classification_utils.dataset_utils import dataset_parser
# from morphological_classification_utils.dataset_utils import remove_invalid_data

from morphological_classification_utils.dataset_utils import dataset_parser
from morphological_classification_utils.dataset_utils import remove_invalid_data

class Classifier:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        params = dataset_parser(self.train_dataset)
        self.c_list = params['c_list']
        self.a_list = params['a_list']
        self.m_list = params['m_list']
        self.continue_th = 0.1

        self.reset()

    def reset(self):
        self.X = list()  # タスク中に得たデータを格納する
        self.G = list()  # 終了判定に使われる

        # 事後分布のn回目の分母分子を分けて保存。
        # 数値誤差対策。
        self.post_prob_list = dict()
        for c in self.c_list:
            self.post_prob_list[c] = dict()
            self.post_prob_list[c]['numerator'] = list()
            self.post_prob_list[c]['denominator'] = list()
            self.post_prob_sum = None

    def should_continue(self):
        if len(self.G) < 2:
            return True
        
        delta = self.G[-1] - self.G[-2]

        if delta > self.continue_th:
            return True
        else:
            return False

    def calc_G(self):
        g = 0
        for c in self.c_list:
            prir_p = self.prior_prob(c)
            post_p = self.posterior_prob(c)

            g_elem = post_p * np.log(post_p/prir_p)

            # np.logで負の無限大になってしまう対策
            if np.isnan(g_elem):
                g += 0
            else:
                g += g_elem


        # 正規化
        # これがないと、クラス数によってGが変動する。
        g /= len(self.c_list)

        self.G.append(g)
 
    def append_x(self, x : list):
        self.X.append(x)

    def calc_next_action_morphology(self):

        next_a = None
        next_m = None
        max_benefit = 0

        for a, m in product(self.a_list, self.m_list):
            unbiased_benefit = self.unbiased_benefit_estimation(a, m)
            biased_benefit   = self.to_biased_benefit(unbiased_benefit)
            
            if biased_benefit > max_benefit:
                next_a = a
                next_m = m
                max_benefit = biased_benefit

        return next_a, next_m, max_benefit

    def estimate_c(self):
        max_c = self.c_list[0]
        max_prob = 0

        for c in self.c_list:
            prob = self.posterior_prob(c)
            if prob > max_prob:
                max_prob = prob
                max_c = c

        return max_c, max_prob

    def prior_prob(self, c):
        return 1 / len(self.c_list)

    def posterior_prob(self, c):
        prob_c = np.prod(self.post_prob_list[c]['numerator']) / \
            np.prod(self.post_prob_list[c]['denominator']) * self.prior_prob(c)
        
        normalized = self.post_prob_normalizer(prob_c)

        return normalized

    def update_posterior_prob(self):
        a, m, s = self.X[-1]

        # n個目の分母の計算(denominator)
        denominator = 0
        for k in self.c_list:
            dst_key_kam = f'class-{k}__action-{a}__morphology-{m}'
            mean_kam = self.train_dataset[dst_key_kam]['mean']
            var_kam = self.train_dataset[dst_key_kam]['var']
            denominator += stats.norm.pdf(x=s, loc=mean_kam, scale=np.sqrt(var_kam)) *\
                self.prior_prob(k)

        for c in self.c_list:    
            # n個目の分子の計算(numerator)
            dst_key_cam = f'class-{c}__action-{a}__morphology-{m}'
            mean_cam = self.train_dataset[dst_key_cam]['mean']
            var_cam  = self.train_dataset[dst_key_cam]['var']
            numerator = stats.norm.pdf(x=s, loc=mean_cam, scale=np.sqrt(var_cam))
            self.post_prob_list[c]['numerator'].append(numerator)
            self.post_prob_list[c]['denominator'].append(denominator)

        # 正規化のための関数を作製（事後分布の和が１になるようにする関数）
        sum = 0
        for c in self.c_list:
            prob_c = np.prod(self.post_prob_list[c]['numerator']) / \
                np.prod(self.post_prob_list[c]['denominator']) * self.prior_prob(c)
            sum += prob_c


        self.post_prob_normalizer = lambda post_prob : post_prob / sum

    def to_biased_benefit(self, unbiased):
        biased = unbiased
        return biased

    def unbiased_benefit_estimation(self, a, m):

        if len(self.X) == 0:
            prob = lambda c: self.prior_prob(c)
        else:
            prob = lambda c: self.posterior_prob(c)
            
        benefit = 0
        for c in self.c_list:
            denominator = 0
            for k in self.c_list:
                denominator += self.bhattacharyya_distance(c,k,a,m)*prob(k)
            benefit += (prob(c))**2 / denominator

        return benefit


    def bhattacharyya_distance(self, c_1, c_2, a, m):
        mean_1, var_1 = self.get_mean_var(c_1, a, m)
        mean_2, var_2 = self.get_mean_var(c_2, a, m)
        
        if var_1 == 0 or var_2 == 0:
            raise ValueError('Var is zero. bhattacharayya distance Cannot be calculated.')
        
        return np.sqrt(2*np.sqrt(var_1)*np.sqrt(var_2)/(var_1 + var_2))*\
            np.exp(-1 * (mean_1 - mean_2)**2 / (4*(var_1 + var_2)))

    def get_mean_var(self, c, a, m):
        train_dataset_key = f'class-{c}__action-{a}__morphology-{m}'
        mean = self.train_dataset[train_dataset_key]['mean']
        var  = self.train_dataset[train_dataset_key]['var']
        return mean, var

    

def arg_parser(argv):
    parser = argparse.ArgumentParser(description='Controller Class')
    parser.add_argument('-d', '--dataset_file_path', type=str,
                        default='C:\\Users\\toshi\\OneDrive\\Documents\\OsakaUniversity\\Hosodalab\\projects\\2020\\master_thesis\\dataset\\20201119123639.pickle')
    args = parser.parse_args()
    return args

def _main():
    args = arg_parser(sys.argv)

    with open(args.dataset_file_path, 'rb') as f:
        train_dataset = pickle.load(f)

    train_dataset = remove_invalid_data(train_dataset)

    controller = Controller(
        train_dataset=train_dataset)

    next_a, next_m, max_belief =\
        controller.calc_next_action_morphology()
    
    print(next_a, next_m, max_belief)

if __name__=='__main__':
    _main()
