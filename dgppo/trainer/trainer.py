import wandb
import os
import numpy as np
import jax
import jax.random as jr
import functools as ft
import jax.numpy as jnp
import json # Added for saving best model info

from time import time
from tqdm import tqdm

from .data import Rollout
from .utils import test_rollout
from ..env import MultiAgentEnv
from ..algo.base import Algorithm
from jax.random import PRNGKey


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: Algorithm,
            gamma: float,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True,
            initial_step: int = 0 # New argument for resuming
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.gamma = gamma
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True) # Use makedirs for parent dirs too
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True) # Use makedirs for parent dirs too

        wandb.login()
        # resume="allow" is important for WandB to correctly resume the run
        wandb.init(name=params['run_name'], project='dgppo', group=env.__class__.__name__, dir=self.log_dir, resume="allow")

        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']

        # Initialize steps and PRNG key based on initial_step
        self.update_steps = initial_step
        self.global_step_counter = initial_step # This will be the loop counter
        self.key = jax.random.PRNGKey(seed)

        # Advance the PRNG key to the correct state if resuming
        if initial_step > 0:
            print(f"Advancing JAX PRNG key by {initial_step} steps for resumption...")
            for _ in range(initial_step):
                _, self.key = jax.random.split(self.key)
            print("JAX PRNG key advanced.")

        # Initialize best model tracking
        self.best_eval_reward = -float('inf') # Track the best reward found so far
        self.best_step = initial_step # Track the step at which the best reward was found

        # If resuming, try to load previous best model info
        best_model_info_path = os.path.join(self.log_dir, 'best_model_info.json')
        if os.path.exists(best_model_info_path):
            try:
                with open(best_model_info_path, 'r') as f:
                    best_info = json.load(f)
                self.best_eval_reward = best_info.get('best_eval_reward', self.best_eval_reward)
                self.best_step = best_info.get('best_step', self.best_step)
                print(f"Loaded previous best model info: best_eval_reward={self.best_eval_reward:.3f} at step={self.best_step}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode best_model_info.json at {best_model_info_path}. Starting best model tracking from scratch.")


        print(f"Trainer initialized. Starting from update_steps={self.update_steps}, global_step_counter={self.global_step_counter}")

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        return True

    def save_trainer_state(self, current_loop_step: int):
        """
        Saves the internal state of the Trainer (update_steps and PRNG key).
        This should be called alongside algo.save().
        """
        # Create a directory for this specific step's checkpoint if it doesn't exist
        step_checkpoint_dir = os.path.join(self.model_dir, str(current_loop_step))
        os.makedirs(step_checkpoint_dir, exist_ok=True)

        state_path = os.path.join(step_checkpoint_dir, 'trainer_state.npz')
        # Save update_steps and the current JAX PRNG key
        np.savez(state_path, update_steps=self.update_steps, key_array=self.key)
        print(f"Trainer state saved to {state_path}")

    def _save_best_model_info(self):
        """
        Saves information about the best performing model to a JSON file.
        """
        best_info = {
            'best_step': int(self.best_step), # Ensure it's a standard int for JSON
            'best_eval_reward': float(self.best_eval_reward) # Ensure it's a standard float for JSON
        }
        best_model_info_path = os.path.join(self.log_dir, 'best_model_info.json')
        with open(best_model_info_path, 'w') as f:
            json.dump(best_info, f, indent=4)
        print(f"Best model info saved: step {self.best_step}, reward {self.best_eval_reward:.3f}")


    def train(self):
        # record start time
        start_time = time()
        
        # preprocess the rollout function
        init_rnn_state = self.algo.init_rnn_state

        chunk_size = getattr(self.algo, 'chunk_size', 1)

        def test_fn_single(params, key):
            act_fn = ft.partial(self.algo.act, params=params)
            return test_rollout(
                self.env_test,
                act_fn,
                init_rnn_state,
                key,
                chunk_size=chunk_size,
            )


        # # preprocess the rollout function
        # init_rnn_state = self.algo.init_rnn_state
        
        # def test_fn_single(params, key: PRNGKey):
        #     init_rnn_state = self.algo.get_initial_state()
        #     key_x0, _ = jr.split(key, 2)
        #     initial_graph = self.env_test.reset(key_x0)
            
        #     # Correctly pass the params to the act function
        #     act_fn = ft.partial(self.algo.act, params=params)

        #     return test_rollout(
        #         self.env_test,
        #         act_fn, # The act_fn now correctly contains the params
        #         init_rnn_state,
        #         key,
        #         stochastic=self.eval_stochastic_actor
        #     )

        # def test_fn_single(params, key):
        #     act_fn = ft.partial(self.algo.act, params=params)
        #     return test_rollout(
        #         self.env_test,
        #         act_fn,
        #         init_rnn_state,
        #         key
        #     )

        test_fn = lambda params, keys: jax.vmap(ft.partial(test_fn_single, params))(keys)
        test_fn = jax.jit(test_fn)

        # set up keys for testing
        test_key = jr.PRNGKey(self.seed)
        assert self.n_env_test <= 1_000, 'n_env_test must be less than or equal to 1_000'
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        # Initialize tqdm progress bar from the current global step
        pbar = tqdm(total=self.steps, initial=self.global_step_counter, ncols=80)

        # Loop starts from the global_step_counter
        for step in range(self.global_step_counter, self.steps + 1):
            # evaluate the algorithm
            if step % self.eval_interval == 0:
                eval_info = {}
                test_rollouts: Rollout = test_fn(self.algo.params, test_keys)
                total_reward = test_rollouts.rewards.sum(axis=-1)
                reward_min, reward_max = total_reward.min(), total_reward.max()
                reward_mean = np.mean(total_reward)
                reward_final = np.mean(test_rollouts.rewards[:, -1])
                # Ensure cost calculation is robust to empty arrays or all zeros
                cost_values = jnp.maximum(test_rollouts.costs, 0.0)
                # Handle cases where cost_values might be all zeros or empty
                if cost_values.size > 0 and jnp.any(cost_values > 0):
                    cost = cost_values.max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                else:
                    cost = 0.0 # No cost incurred
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)
                eval_info = eval_info | {
                    "eval/reward": reward_mean,
                    "eval/reward_final": reward_final,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                }

                # Log individual reward components from the last step of the first test env
                if hasattr(self.env_test, 'get_reward_components'):
                    try:
                        last_graph = jax.tree_util.tree_map(
                            lambda x: x[0, -1], test_rollouts.graph
                        )
                        eval_info |= self.env_test.get_reward_components(last_graph)
                    except Exception as e:
                        tqdm.write(f"Warning: get_reward_components failed: {e}")
                time_since_start = time() - start_time
                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost:8.4f}, '
                                f'unsafe_frac: {unsafe_frac:6.2f}')
                tqdm.write(eval_verbose)
                wandb.log(eval_info, step=self.update_steps) # Use self.update_steps for WandB

                # Check if current reward is the best so far
                if reward_mean > self.best_eval_reward:
                    self.best_eval_reward = reward_mean
                    self.best_step = step # Store the loop step, not update_steps
                    self._save_best_model_info() # Save the best model info

            # save the model and trainer state
            if self.save_log and step % self.save_interval == 0:
                # The algo saves its own state (models and optimizers)
                self.algo.save(os.path.join(self.model_dir), step)
                # The trainer saves its own state (update_steps and PRNG key)
                self.save_trainer_state(step)

            # collect rollouts
            key_x0, self.key = jax.random.split(self.key) # Update self.key for next iteration
            key_x0 = jax.random.split(key_x0, self.n_env_train)
            rollouts = self.algo.collect(self.algo.params, key_x0)

            # update the algorithm
            # The 'step' argument here is crucial for algo's internal scheduling
            update_info = self.algo.update(rollouts, step)
            wandb.log(update_info, step=self.update_steps) # Use self.update_steps for WandB
            self.update_steps += 1 # Increment update_steps after an actual update

            pbar.update(1)

        pbar.close() # Close the progress bar when training is done
        wandb.finish() # Ensure WandB run is properly finished
