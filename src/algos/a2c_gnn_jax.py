"""
A2C-GNN Jax implementation
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import jax
from jax import jit
import jax.numpy as jnp
import jax.random as jrandom
from flax import linen as nn
import optax
import equinox as eqx
from flax.training import train_state, checkpoints, orbax_utils
import orbax
import orbax.checkpoint



import numpy as np 
from torch_geometric.utils import grid
from collections import namedtuple
import os

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## PARSER ###################
#########################################

class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """
    def __init__(self, env, T=10, grid_h=4, grid_w=4, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.grid_h = grid_h
        self.grid_w = grid_w
        
    def parse_obs(self, obs):
        x1 = jnp.array([obs[0][n][self.env.time+1]*self.s for n in self.env.region]).reshape(1, 1, self.env.nregion)
        x2 = jnp.array([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t])*self.s for n in self.env.region] \
                          for t in range(self.env.time+1, self.env.time+self.T+1)]).reshape(1, self.T, self.env.nregion)
        x3 = jnp.array([[sum([(self.env.scenario.demand_input[i,j][t])*(self.env.price[i,j][t])*self.s \
                          for j in self.env.region]) for i in self.env.region] for t in range(self.env.time+1, self.env.time+self.T+1)]).reshape(1, self.T, self.env.nregion)

        x = jnp.concatenate((x1, x2, x3), axis=1).squeeze(0).reshape(21,self.env.nregion).T

        
        edge_index, pos_coord = grid(height=self.grid_h, width=self.grid_w)
        adj_matrix = np.zeros((self.grid_h*self.grid_w, self.grid_h*self.grid_w))
        for e0, e1 in zip(edge_index[0], edge_index[1]):
            adj_matrix[e0,e1] = 1
        adj_matrix = jnp.array(adj_matrix)
        
        return x, adj_matrix
    

#########################################
############ GCNConv_JAX ################
######################################### 
class GCNConv_JAX(nn.Module):
    out_dim : int
    """
    GCNConv from pytorch converted to JAX 
    Layer constructor function for a Graph Convolution layer similar to https://arxiv.org/abs/1609.02907
    """
    @nn.compact
    def __call__(self, x, adj_matrix, **kwargs):
        x = nn.Dense(features=self.out_dim, name='gcn')(x)
        x = jnp.dot(adj_matrix, x)
        
        num_neighbours = jnp.sum(adj_matrix, axis=-1, keepdims=True)
 
        return x/num_neighbours

    
#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """
    in_channels: int
    # def __post_init__(self):
    #     super().__post_init__() 
        
    # def __init__(self, in_channels, out_channels):
        
    #     self.conv1 = GCNConv_JAX(in_channels)
    #     self.lin1 = nn.Dense(32)
    #     self.lin2 = nn.Dense(32)
    #     self.lin3 = nn.Dense(1)


    #     # key1,key2,key3, key4 = jrandom.split(jrandom.PRNGKey(0), num=4)

    #     # self.conv_params = {'gcn': {
    #     #                     'kernel': nn.initializers.glorot_uniform()(key1, (in_channels, in_channels)),
    #     #                     'bias': nn.initializers.zeros(key2, (in_channels,))}}
        
        

    # def __call__(self, x, adj_matrix):
    #     # out = jax.nn.relu(self.conv1.apply({'params': self.conv_params}, x, adj_matrix))
    #     # x = out + x

    #     # key1, key2, key3 = jrandom.split(jrandom.PRNGKey(0), num=3)
    #     # x = jax.nn.relu(self.lin1.apply(self.lin1.init(key1, jnp.ones(x.shape)), x))
    #     # x = jax.nn.relu(self.lin2.apply(self.lin2.init(key2, jnp.ones(x.shape)), x))
    #     # x = self.lin3.apply(self.lin3.init(key3, jnp.ones(x.shape)), x)
    #     # return x

    #     out = jax.nn.relu(self.conv1(x, adj_matrix))
    #     x = out + x
    #     x = jax.nn.relu(self.lin1(x))
    #     x = jax.nn.relu(self.lin2(x))
    #     x = self.lin3(x)
    #     return x

    @nn.compact
    def __call__(self, x, adj_matrix):
        # out = jax.nn.relu(self.conv1.apply({'params': self.conv_params}, x, adj_matrix))
        # x = out + x

        # key1, key2, key3 = jrandom.split(jrandom.PRNGKey(0), num=3)
        # x = jax.nn.relu(self.lin1.apply(self.lin1.init(key1, jnp.ones(x.shape)), x))
        # x = jax.nn.relu(self.lin2.apply(self.lin2.init(key2, jnp.ones(x.shape)), x))
        # x = self.lin3.apply(self.lin3.init(key3, jnp.ones(x.shape)), x)
        # return x

        out = jax.nn.relu(GCNConv_JAX(self.in_channels)(x, adj_matrix))
        x = out + x
        x = jax.nn.relu(nn.Dense(32)(x))
        x = jax.nn.relu(nn.Dense(32)(x))
        x = nn.Dense(1)(x)
        return x


#########################################
############## CRITIC ###################
#########################################

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    in_channels: int
    
    # def __init__(self, in_channels, out_channels):
    #     super().__init__()
        
    #     self.conv1 = GCNConv_JAX(in_channels)
    #     self.lin1 = nn.Dense(32)
    #     self.lin2 = nn.Dense(32)
    #     self.lin3 = nn.Dense(1)

    #     key1,key2 = jrandom.split(jrandom.PRNGKey(0))

    #     self.conv_params = {'gcn': {
    #                         'kernel': nn.initializers.glorot_uniform()(key1, (in_channels, in_channels)),
    #                         'bias': nn.initializers.zeros(key2, (in_channels,))}}


    
    # def __call__(self, x, adj_matrix):
    #     out = jax.nn.relu(self.conv1.apply({'params': self.conv_params}, x, adj_matrix))
    #     x = out + x 
    #     x = jnp.sum(x, axis=0)

    #     key1, key2, key3 = jrandom.split(jrandom.PRNGKey(0), num=3)
    #     x = jax.nn.relu(self.lin1.apply(self.lin1.init(key1, jnp.ones(x.shape)), x))
    #     x = jax.nn.relu(self.lin2.apply(self.lin2.init(key2, jnp.ones(x.shape)), x))
    #     x = self.lin3.apply(self.lin3.init(key3, jnp.ones(x.shape)), x)
    #     return x
    
    @nn.compact
    def __call__(self, x, adj_matrix):
        out = jax.nn.relu(GCNConv_JAX(self.in_channels)(x, adj_matrix))
        x = out + x
        x = jnp.sum(x, axis=0)
        x = jax.nn.relu(nn.Dense(32)(x))
        x = jax.nn.relu(nn.Dense(32)(x))
        x = nn.Dense(1)(x)
        return x

#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    
    def __init__(self, env, input_size, eps=np.finfo(np.float32).eps.item()):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size

        self.obs_parser = GNNParser(self.env)
        
        # Initialize models and params 
        key1, key2, key3 = jrandom.split(jrandom.PRNGKey(0), 3)
        inp_x = jnp.ones((16,self.input_size))
        inp_adj = jnp.ones((16,16))

        self.actor = GNNActor(self.input_size)
        self.actor_params = self.actor.init(key2, inp_x, inp_adj)

        self.critic = GNNCritic(self.input_size)
        self.critic_params = self.critic.init(key3, inp_x, inp_adj)
        
        
        # Configure optimizers
        self.optimizer = optax.adam(1e-4)
        # opt_states = dict()
        # opt_states['a_state'] = self.optimizer.init(eqx.filter(self.actor, eqx.is_inexact_array))
        # opt_states['c_state'] = self.optimizer.init(eqx.filter(self.critic, eqx.is_inexact_array))
        # self.opt_states = opt_states

        self.state_a = train_state.TrainState.create(apply_fn=self.actor.apply,
                                            params=self.actor_params,
                                            tx=self.optimizer)
                        
        self.state_c = train_state.TrainState.create(apply_fn=self.critic.apply,
                                            params=self.critic_params,
                                            tx=self.optimizer)
        
        self.loss_actor = []
        self.loss_critic = []
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        
    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x, adj_matrix = self.parse_obs(obs)
        
        # actor: computes concentration parameters of a Dirichlet distribution        
        # a_out = self.actor.apply(self.actor_params, x, adj_matrix)
        a_out = self.state_a.apply_fn(self.actor_params, x, adj_matrix)
        concentration = jax.nn.softplus(a_out).reshape(-1) + jitter

        # critic: estimates V(s_t)
        # value = self.critic.apply(self.critic_params, x, adj_matrix)
        value = self.state_c.apply_fn(self.critic_params, x, adj_matrix)
        return concentration, value
    
    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    def select_action(self, obs):
        concentration, value = self.forward(obs)
        
        action = jrandom.dirichlet(jrandom.PRNGKey(0),concentration)

        self.saved_actions.append(SavedAction(jax.scipy.stats.dirichlet.logpdf(action, concentration), value[0]))

        return list(action)

    def smooth_l1_loss_jax(self, model, params, x, y, beta=1.0):
        diff = jnp.abs(x - y)
        smooth_l1 = jnp.where(diff < beta, (0.5 * diff**2)/beta, (diff - 0.5 * beta))
        return jnp.sum(smooth_l1.mean())

    def policy_loss(self, model, params, log_probs, values, returns):
        advantage = returns - values
        return jnp.sum(-log_probs * advantage)

    def training_step(self, step):
        R = 0
        saved_actions = self.saved_actions
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = jnp.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        saved_actions_arr = np.array(saved_actions, dtype=[('log_prob', float), ('value', float)])
        log_probs = saved_actions_arr['log_prob']
        values = saved_actions_arr['value']

        # Gradient and loss
        # loss_a, grad_a = eqx.filter_value_and_grad(self.policy_loss)(self.actor, log_probs,values, returns)
        # a_update, self.opt_states['a_state'] = self.optimizer.update(grad_a, self.opt_states['a_state'])
        # self.actor = eqx.apply_updates(self.actor, a_update)

        # loss_c, grad_c = eqx.filter_value_and_grad(self.smooth_l1_loss_jax)(self.critic, values,returns)
        # c_update, self.opt_states['c_state'] = self.optimizer.update(grad_c, self.opt_states['c_state'])
        # self.critic = eqx.apply_updates(self.critic, c_update)

        loss_a, grad_a = jax.value_and_grad(self.policy_loss, argnums=1)(self.state_a, self.state_a.params, log_probs, values, returns)
        self.state_a = self.state_a.apply_gradients(grads=grad_a)
        self.loss_actor.append(loss_a)


        loss_c, grad_c = jax.value_and_grad(self.smooth_l1_loss_jax, argnums=1)(self.state_c, self.state_c.params, values, returns)
        self.state_c = self.state_c.apply_gradients(grads=grad_c)
        self.loss_critic.append(loss_c)
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
    

    def save_checkpoint(self, step, path='ckpt.pth'):

        checkpoints.save_checkpoint(ckpt_dir=path,  # Folder to save checkpoint in
                            target=self.state_a,  # What to save. To only save parameters, use model_state.params
                            step = step,
                            prefix='actor_gnn_test',  # Checkpoint file name prefix
                            overwrite=True   # Overwrite existing checkpoint files
                           )
        
        checkpoints.save_checkpoint(ckpt_dir=path,  # Folder to save checkpoint in
                            target=self.state_c,  # What to save. To only save parameters, use model_state.params
                            step = step,
                            prefix='critic_gnn_test',  # Checkpoint file name prefix
                            overwrite=True   # Overwrite existing checkpoint files
                            )

        # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # save_args = orbax_utils.save_args_from_target(self.actor_params)
        # orbax_checkpointer.save(os.path.join(path, f'actor_gnn_test{step}.ckpt'), self.actor_params, save_args=save_args)


    def load_checkpoint(self, path='ckpt.pth'):
        # self.state_a = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(path, f'actor_gnn_test29'), target=None) 
        # self.state_c = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(path, f'critic_gnn_test29'), target=None)


        self.state_a = checkpoints.restore_checkpoint(
                                             ckpt_dir=path,
                                             target=self.state_a, 
                                             prefix='actor_gnn_test29')


        self.state_c = checkpoints.restore_checkpoint(
                                             ckpt_dir=path,
                                             target=self.state_c, 
                                             prefix='critic_gnn_test29')
        
        



