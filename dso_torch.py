import numpy as np 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from networks import Actor
from torch.distributions.categorical import Categorical
from utilities import Node
import math
import scipy.optimize as spo

class Dso():
    def __init__(self, X, y, network_dims, batch_size, num_layers, lr, tokens_lib, tokens_indices, constraints):
    

        self.tokens_lib = tokens_lib
        self.tokens_indices = tokens_indices
        self.constraints = constraints
        
        
        self.network_dims = network_dims
        
        # Risk Factor
        self.epsilon = .2
        # Entropy coefficient
        self._lambda = .3
        
        self.min_sequence_length = 3
        self.max_sequence_length = 10
        
        self.best_expression = None
        self.best_expression_reward = None    
        self.best_expression_constants = None

        self.batch_size = batch_size
        self.n_expressions = 10
        
        self.X = X
        self.y = y
        
        
        
        self.actor = Actor(lr, network_dims[0], network_dims[1], num_layers, network_dims[-1])        
    
    def arity(self, token_index):
        return self.tokens_lib[self.tokens_indices[token_index]]
    
    def build_tree(self, traversal):
        root = Node(traversal[0].item())
        i = self.append_node(traversal, root, 1)
        return root
        
        
    def append_node(self, traversal, node, i):
        if self.arity(node.data) == 0 or i >= len(traversal):
            return i
        
        elif self.arity(node.data) == 1:
            node.left = Node(traversal[i].item())
            i = self.append_node(traversal, node.left, i+1)
        
        elif self.arity(node.data) == 2:
            node.left = Node(traversal[i].item())
            i = self.append_node(traversal, node.left, i+1)
            
            node.right = Node(traversal[i].item())
            i = self.append_node(traversal, node.right, i+1)
        return i

    
    def evaluate_expression_tree(self, root, values, c_values):
        # empty tree
        if root is None:
            return 0, c_values
        # leaf node
        if root.left is None and root.right is None:
            if self.tokens_indices[root.data].startswith('x'):
                return values[int(self.tokens_indices[root.data][2:])], c_values
            elif self.tokens_indices[root.data].startswith('c'):
                value = c_values[0]
                c_values = c_values[1:]
                return value, c_values
            # return float(root.data)
     
        # evaluate left tree
        left_sum, c_values = self.evaluate_expression_tree(root.left, values, c_values)
        # print('left: ', left_sum)
        # evaluate right tree
        right_sum, c_values = self.evaluate_expression_tree(root.right, values, c_values)
        # print('right', right_sum)
        # check which operation to apply       88888888888888888888888888888888888888
        if self.tokens_indices[root.data] == '+':
            return left_sum + right_sum, c_values
     
        elif self.tokens_indices[root.data] == '-':
            return left_sum - right_sum, c_values
     
        elif self.tokens_indices[root.data] == '*':
            return left_sum * right_sum, c_values
        
        elif self.tokens_indices[root.data] == '/':
            if right_sum == 0:
                print('wrooooong1 #####################################################')
            return (left_sum / right_sum, c_values) if np.abs(right_sum) > 0.001 else (1., c_values)
        
        elif self.tokens_indices[root.data] == 'sin':
            return math.sin(left_sum), c_values
        
        elif self.tokens_indices[root.data] == 'exp':
            # print('wrooooong3 #####################################################')
            return math.exp(left_sum), c_values
        
        elif self.tokens_indices[root.data] == 'log':
            # if left_sum <= 0:
            #     print('wrooooong2 #####################################################')
            # if np.abs(left_sum) > 0.001:
            #     print('HI')
            # a = np.log(np.abs(left_sum))
            # b = np.where(np.abs(left_sum) > 0.001, [np.log(np.abs(left_sum)), 0.])
            return (np.log(np.abs(left_sum)), c_values) if np.abs(left_sum) > 0.001 else (0. , c_values)
        
        elif self.tokens_indices[root.data] == 'cos':
            return math.cos(left_sum), c_values
        else:
            print('nothing ###################################################################')

    def compute_rewards(self, y_pred, y):
        mse = nn.MSELoss()
        loss = T.sqrt(mse(y_pred, y))
        std = T.std(y)
        loss = loss / std
        reward = 1/(1+loss)
        return reward
    
    
    def optimize_constants(self, tau, root):
        def f(c):
            # Get predictions
            y_pred = []
            for value in self.X:
                y_pred.append(self.evaluate_expression_tree(root, value, c)[0])
            # Compute reward
            reward = self.compute_rewards(T.tensor(y_pred), T.tensor(self.y))
            return T.multiply(T.tensor(-1), reward).cpu().detach().numpy()
    
        # occ = T.bincount(tau)
        c_occ = np.count_nonzero(tau.numpy() == 9)
        # print(tau)
        # print(c_occ)
        if c_occ > 0:  
            c_start = np.zeros(c_occ)        
            tau_c = spo.minimize(f, c_start, method='BFGS', tol=1e-6).x
        else:
            tau_c = np.array([])
        return tau_c
            
    
    
    
    def parent_sibling(self, tau):
        if tau.shape[0] == 0:
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            return None, None
            
        counter = 0
        if self.arity(tau[-1].item()) > 0:
            parent = tau[-1]
            sibling = None
            # print(T.tensor([int(parent.numpy())]))
            # print(T.tensor([1]))
            return parent, sibling
        for i in range(tau.shape[0]-1, -1, -1):
            counter += self.arity(tau[i].item()) - 1
            if counter == 0:
                parent = tau[i]
                sibling = tau[i+1]
                # print('popppppp:', parent.item())
                return parent, sibling
            # print('counter:', counter)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@hii')
        return None, None
            
    # def get_tree_depth(self, root, node):
        
        
    def vaiolate_constraint(self, tau, tau_index, counter, iteration):
        token = self.tokens_indices[tau_index]
        parent, sibling = self.parent_sibling(tau)
        
        # print('parent: ', parent)
        if parent is not None:
            
            key = self.tokens_indices[parent.item()]
        else:
            key = 'no_parent'
        
        # if iteration > self.max_sequence_length / 2 - 2:
        #     key = 'max_level'
        
        # print(f'left: {counter + self.arity(tau_index) - 1} -- right {self.max_sequence_length - iteration - 1}')
        if counter + self.arity(tau_index) - 1 > self.max_sequence_length - iteration - 1:
            # print('ppsssssssssssstt')
            return True
        
        if key in self.constraints.keys() and token in self.constraints[key]:
            # print('parent: ', key)
            # print('child: ', token)
            return True  
        else:
            # print('gooooooooooood')
            return False
        
    def apply_constraints(self, psi, tau, counter, iteration):
        for i in range(len(self.tokens_indices)):
            if self.vaiolate_constraint(tau, i, counter, iteration):
                psi[i] = 0.0
                # print(psi)
                
        psi = psi / psi.sum()
        # print('psi: ', psi)
        return psi
    
    def sample_expression(self):
        # Initialize empty traversal
        tau = T.tensor([])
        tau_prob = T.ones((1))
        tau_entropy = T.zeros((1))

        # Initialize counter for number of unselected nodes 
        counter = 1
        # Initial RNN input is empty parent and sibling
        parent_sibling = T.zeros((1, 1, self.network_dims[0]))
        # Initialize RNN cell state to zero
        # (h, c) = (None, None)
        h = None
        
        for i in range(self.max_sequence_length):
            # Emit probabilities; update state
            # print(parent_sibling)
            psi, h = self.actor.forward(parent_sibling, h)
            psi = psi.squeeze()
            # Adjust probabilities
            # print(psi)
            psi = self.apply_constraints(psi, tau, counter, i)
            # print(psi)
            # Sample next token
            dist = Categorical(psi)
            token_index = dist.sample()
            # print(token_index)
            token_prob = psi[token_index]
            # Append token to traversal
            # print(token_index.reshape(-1))
            tau = T.cat((tau, token_index.reshape(-1)))
            tau_prob = T.multiply(tau_prob, token_prob)
            tau_entropy += T.multiply(T.sum(T.multiply(psi, T.log(psi+1e-18))), -1)
            # print(tau_entropy)
            # Update number of unselected nodes
            counter += self.arity(token_index.item()) - 1
            # If expression is complete, return it
            if counter == 0:
                for i in range(self.max_sequence_length - tau.shape[0]):
                    tau = T.cat((tau, T.tensor([-1])), dim=0)
                return tau.type(T.int64), tau_prob, tau_entropy
            # Compute next parent and sibling
            # print('##########################################')
            # a = tau.numpy().astype('int')
            # tokens = []
            # for i in a:  
            #     tokens.append(self.tokens_indices[i])
            # print(tokens)
            # print('##########################################')
            parent, sibling = self.parent_sibling(tau)
            item_length = int(self.network_dims[0]/2)
            if not sibling:
                parent_sibling = T.cat((F.one_hot(parent.type(T.int64), item_length).view(1, 1, -1), T.zeros(1, 1, item_length, dtype=T.int64)), dim=-1).type(T.float32)
            else:
                parent_sibling = T.cat((F.one_hot(parent.type(T.int64), item_length).view(1, 1, -1), F.one_hot(sibling.type(T.int64), item_length).view(1, 1, -1)), dim=-1).type(T.float32)
            # print(counter)
        return None, None, None
        
    
    # See if the resulted expressions are valid
    
    def train(self, batch):
        # Sample N expressions
        Tau = T.empty((0, self.max_sequence_length), dtype=T.int32)
        Tau_probs = T.empty((0, 1))
        Tau_entropy = T.empty((0, 1))
        for i in range(self.n_expressions):
            tau, tau_probs, tau_entropy = self.sample_expression()
            # print(tau)
            tokens = []
            for i in tau:  
                if int(i) in self.tokens_indices.keys():
                    tokens.append(self.tokens_indices[int(i)])
            print(tokens)
            
            Tau = T.cat((Tau, tau.unsqueeze(0)), dim=0)
            Tau_probs = T.cat((Tau_probs, tau_probs.unsqueeze(0)), dim=0)
            Tau_entropy = T.cat((Tau_entropy, tau_entropy.unsqueeze(0)), dim=0)
            
                
        
        Tau_roots = []
        Tau_constants = []
        for i in range(self.n_expressions):
            # Build tree
            tau = Tau[i][Tau[i]!=-1]
            # print(tau)
            Tau_roots.append(self.build_tree(tau))
            # Optimize constant with respect to the reward function
            Tau_constants.append(self.optimize_constants(tau, Tau_roots[i]))
        
        # Compute rewards
        # -- compute y_pred from x
        y_pred = []
        for i in range(self.n_expressions):
            y = []
            for j in range(len(self.X)):
                y.append(self.evaluate_expression_tree(Tau_roots[i], self.X[j], Tau_constants[i])[0])
            y_pred.append(y)
        # -- get rewards
        rewards = T.empty(0)
        for i in range(self.n_expressions):
            a  = self.compute_rewards(T.tensor(y_pred[i]), T.tensor(self.y)).unsqueeze(0)
            rewards = T.cat((rewards, a), dim=0)
            
        # Compute rewards threshold 
        reward_threshold = T.quantile(rewards, (1 - self.epsilon), dim=0, keepdim=True)
        
        # Select subset of expressions above threshold and their corresponding subset of rewards
        Tau_best = T.empty(0, self.max_sequence_length)
        rewards_best = T.empty(0)
        for i in range(self.n_expressions):
            if rewards[i] > reward_threshold:
                rewards_best = T.cat((rewards_best, T.tensor([rewards[i]])), dim=0)
                Tau_best = T.cat((Tau_best, Tau[i].unsqueeze(0)), dim=0) 
        
        # Compute risk-seeking policy loss
        loss_g1 = T.mean(T.multiply((rewards_best - reward_threshold), T.log(Tau_probs)))
        # Compute entropy loss
        loss_g2 = T.mean(T.multiply(T.tensor(self._lambda), Tau_entropy))
        # Compute loss
        loss = T.multiply((loss_g1 + loss_g2), -1) 
        # Update the actor
        T.autograd.set_detect_anomaly(True)
        self.actor.optimizer.zero_grad()
        T.autograd.backward(loss)
        self.actor.optimizer.step()
        # Update best expression
        self.best_expression_reward = T.max(rewards_best)
        self.best_expression = Tau_best[T.argmax(rewards_best)]
        self.best_expression_constants = Tau_constants[T.argmax(rewards_best)]
        
        
        