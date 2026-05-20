import argparse
import datetime
import os
import ipdb
import numpy as np
import wandb
import yaml
import glob # Added for finding latest checkpoint
import json # Added for loading best model info

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.trainer import Trainer
from dgppo.trainer.utils import is_connected


def train(args):
    print(f"> Running train.py {args}")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    # create environments (these are created regardless of resume/new run)
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
    )

    # --- Check for checkpoint loading and determine run parameters ---
    loaded_checkpoint_path = None
    initial_step = 0 # This will store the step from which we are resuming

    if args.load_checkpoint:
        # If a specific checkpoint path is provided, use it directly
        if os.path.exists(args.load_checkpoint) and os.path.isdir(args.load_checkpoint):
            loaded_checkpoint_path = args.load_checkpoint
            print(f"Attempting to load checkpoint from: {loaded_checkpoint_path}")
        else:
            print(f"Error: Checkpoint path '{args.load_checkpoint}' not found or is not a directory.")
            print("Starting new training run.")
            args.load_checkpoint = None # Reset to start new run
    elif args.resume_last:
        # Try to find the latest run in the log directory for the given env/algo/seed
        search_path = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_*"
        potential_runs = sorted(glob.glob(search_path), key=os.path.getmtime, reverse=True)
        if potential_runs:
            latest_run_dir = potential_runs[0]
            # Verify if this directory contains a 'models' subdirectory and a 'config.yaml'
            if os.path.exists(os.path.join(latest_run_dir, "models")) and \
               os.path.exists(os.path.join(latest_run_dir, "config.yaml")):
                loaded_checkpoint_path = latest_run_dir
                print(f"Resuming from latest run: {loaded_checkpoint_path}")
            else:
                print(f"Latest run directory '{latest_run_dir}' is not a valid checkpoint (missing models/config.yaml).")
                print("Starting new training run.")
        else:
            print(f"No previous runs found for env={args.env}, algo={args.algo}, seed={args.seed}.")
            print("Starting new training run.")

    # Determine log_dir and run_name based on whether we are resuming or starting new
    if loaded_checkpoint_path:
        # --- Resuming an existing run ---
        log_dir = loaded_checkpoint_path

        # Load original config to ensure consistent algo initialization
        try:
            with open(os.path.join(log_dir, "config.yaml"), "r") as f:
                loaded_config = yaml.safe_load(f)
            # Update args with loaded config, but allow new CLI arguments to override
            for key, value in loaded_config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
            # For parameters that are flags (action='store_true'), ensure they are correctly overridden
            # by CLI args if present in loaded_config but not in args
            for flag_arg in ['debug', 'full_observation', 'no_cbf_schedule', 'cost_schedule', 'no_rnn', 'use_lstm']:
                if flag_arg in loaded_config and not hasattr(args, flag_arg):
                    setattr(args, flag_arg, loaded_config[flag_arg])


            # Determine the step to load (best vs. latest)
            if args.load_best:
                best_model_info_path = os.path.join(loaded_checkpoint_path, 'best_model_info.json')
                if os.path.exists(best_model_info_path):
                    with open(best_model_info_path, 'r') as f:
                        best_info = json.load(f)
                    initial_step = best_info.get('best_step', 0)
                    print(f"Loading BEST model found at step: {initial_step} (reward: {best_info.get('best_eval_reward'):.3f})")
                else:
                    print(f"Warning: best_model_info.json not found in {loaded_checkpoint_path}. Falling back to latest step.")
                    # Fallback to latest if best_model_info.json is missing
                    model_path_in_checkpoint = os.path.join(loaded_checkpoint_path, "models")
                    if os.path.exists(model_path_in_checkpoint):
                        models_subdirs = [d for d in os.listdir(model_path_in_checkpoint) if d.isdigit()]
                        if models_subdirs:
                            initial_step = max([int(s) for s in models_subdirs])
                            print(f"Found latest model at step: {initial_step}")
                        else:
                            initial_step = 0
                            print("No numbered model directories found. Starting from step 0.")
            else: # Not loading best, so load the latest step
                model_path_in_checkpoint = os.path.join(loaded_checkpoint_path, "models")
                if os.path.exists(model_path_in_checkpoint):
                    models_subdirs = [d for d in os.listdir(model_path_in_checkpoint) if d.isdigit()]
                    if models_subdirs:
                        initial_step = max([int(s) for s in models_subdirs])
                        print(f"Found latest model at step: {initial_step}")
                    else:
                        initial_step = 0
                        print("No numbered model directories found. Starting from step 0.")


            # Append a resume tag to the run name for clarity in WandB/logs
            # Try to parse original run_name from log_dir name
            original_run_name_part = os.path.basename(loaded_checkpoint_path)
            run_name = f"RESUME_STEP{initial_step}_{original_run_name_part}"
            if args.name is not None: # Add user-defined name if present
                run_name = f"{args.name}_{run_name}"

        except FileNotFoundError:
            print(f"config.yaml not found in {loaded_checkpoint_path}. Treating as new run.")
            loaded_checkpoint_path = None # Fallback to new run logic
        except yaml.YAMLError as e:
            print(f"Error parsing config.yaml in {loaded_checkpoint_path}: {e}. Treating as new run.")
            loaded_checkpoint_path = None # Fallback to new run logic

    if not loaded_checkpoint_path:
        # --- Starting a brand new run ---
        initial_step = 0 # Ensure initial_step is 0 for new runs

        # Generate a 4 letter random identifier for the run.
        rng_ = np.random.default_rng()
        rand_id = "".join([chr(rng_.integers(65, 91)) for _ in range(4)])

        # set up logger
        start_time_dt = datetime.datetime.now()
        start_time_str = start_time_dt.strftime("%m%d%H%M%S")
        
        # Ensure base log directory exists
        base_log_path = f"{args.log_dir}/{args.env}/{args.algo}"
        if not args.debug and not os.path.exists(base_log_path):
            os.makedirs(base_log_path, exist_ok=True)
        
        # Ensure unique log directory name
        log_dir_suffix = f"seed{args.seed}_{start_time_str}_{rand_id}"
        log_dir = os.path.join(base_log_path, log_dir_suffix)
        while os.path.exists(log_dir): # In case of extremely fast successive runs
            start_time_str = (datetime.datetime.now() + datetime.timedelta(seconds=1)).strftime("%m%d%H%M%S")
            log_dir_suffix = f"seed{args.seed}_{start_time_str}_{rand_id}"
            log_dir = os.path.join(base_log_path, log_dir_suffix)

        run_name = "{}_{}_seed{:03}_{}_{}".format(args.env, args.algo, args.seed, start_time_str, rand_id)
        if args.name is not None:
            run_name = "{}_{}".format(args.name, run_name)

    # create algorithm (uses 'args' which might have been updated from loaded_config)
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=args.cost_weight,
        cbf_weight=args.cbf_weight,
        actor_gnn_layers=args.actor_gnn_layers,
        Vl_gnn_layers=args.Vl_gnn_layers,
        Vh_gnn_layers=args.Vh_gnn_layers,
        rnn_layers=args.rnn_layers,
        lr_actor=args.lr_actor,
        lr_Vl=args.lr_Vl,
        lr_Vh=args.lr_Vh,
        max_grad_norm=2.0,
        alpha=args.alpha,
        cbf_eps=args.cbf_eps,
        seed=args.seed,
        batch_size=args.batch_size,
        use_rnn=not args.no_rnn, # Note: if no_rnn is True, use_rnn becomes False
        use_lstm=args.use_lstm,
        coef_ent=args.coef_ent,
        rnn_step=args.rnn_step,
        gamma=0.99, # This was hardcoded in your original, keep it so
        clip_eps=args.clip_eps,
        lagr_init=args.lagr_init,
        lr_lagr=args.lr_lagr,
        train_steps=args.steps, # This is the total steps for the *new* run
        cbf_schedule=not args.no_cbf_schedule, # Note: if no_cbf_schedule is True, cbf_schedule becomes False
        cost_schedule=args.cost_schedule,
        chunk_size=args.chunk_size,
    )

    # Load algorithm state if resuming
    if initial_step > 0:
        model_load_path = os.path.join(log_dir, 'models') # models are saved inside log_dir/models
        if os.path.exists(model_load_path):
            algo.load(model_load_path, initial_step)
            print(f"Algorithm state loaded from {os.path.join(model_load_path, str(initial_step))}")
        else:
            print(f"Warning: Model directory {model_load_path} not found for loading. Starting with fresh algo.")

    # get training parameters for Trainer initialization
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval
    }

    # create trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        gamma=0.99, # This was hardcoded in your original, keep it so
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
        initial_step=initial_step # Pass the initial step to the Trainer
    )

    # Prepare unified config dictionary for saving
    unified_config = vars(args).copy() # Start with all argparse arguments
    unified_config.update(algo.config) # Add/override with algo-specific config

    # save config (for new runs) or update wandb config (for all runs)
    wandb.config.update(unified_config) # Update WandB with the unified config
    if not args.debug and not loaded_checkpoint_path: # Only dump config for truly new runs
        os.makedirs(log_dir, exist_ok=True) # Ensure log_dir exists before writing config
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(unified_config, f) # Dump the unified config

    # start training
    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("-n", "--num-agents", type=int, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--obs", type=int, required=True)

    # custom arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cost-weight", type=float, default=0.)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument('--full-observation', action='store_true', default=False)
    parser.add_argument('--clip-eps', type=float, default=0.25)
    parser.add_argument('--lagr-init', type=float, default=0.5)
    parser.add_argument('--lr-lagr', type=float, default=1e-7)
    parser.add_argument("--cbf-weight", type=float, default=1.0)
    parser.add_argument("--cbf-eps", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--no-cbf-schedule", action="store_true", default=False)
    parser.add_argument("--cost-schedule", action="store_true", default=False)
    parser.add_argument("--no-rnn", action="store_true", default=False)

    # NN arguments
    parser.add_argument("--actor-gnn-layers", type=int, default=2)
    parser.add_argument("--Vl-gnn-layers", type=int, default=2)
    parser.add_argument("--Vh-gnn-layers", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-Vl", type=float, default=1e-3)
    parser.add_argument("--lr_Vh", type=float, default=1e-3) # Corrected typo from previous version (was lr-Vh)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--coef-ent", type=float, default=1e-2)
    parser.add_argument("--rnn-step", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Action chunking: policy outputs chunk_size actions, executed open-loop")

    # default arguments
    parser.add_argument("--n-env-train", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)

    # Resumption arguments
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help="Path to a specific checkpoint directory to load (e.g., ./logs/env/algo/run_name).")
    parser.add_argument('--resume-last', action='store_true',
                        help="Resume from the last saved checkpoint for the given env/algo/seed.")
    parser.add_argument('--load-best', action='store_true',
                        help="When resuming, load the model from the best performing epoch/step based on eval/reward.")


    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
