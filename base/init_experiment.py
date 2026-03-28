# Experiment initialization function responsible for setting random seeds, configuring experiment directories, and automatically adjusting learning rates
# Main functionalities include:
# 1. Setting a fixed random seed when reproduction mode is enabled
# 2. Creating experiment result storage directory
# 3. Automatically adjusting learning rates based on GPU count, batch size, and gradient accumulation (Since we do each experiment on single GPU, we don't use lr scaling and gradient accumulation in practice, the stability of this part need to double check)
# 4. Handling learning rate scaling for different computing environments (CPU/GPU)
import os
import pytorch_lightning as pl
import shutil
import datetime
import sys
def initExperiment(opts):
    if opts.reproduce:
        pl.seed_everything(42, workers=True) # set the random seed to 42 
        opts.deterministic = opts.reproduce
        opts.benchmark = not opts.reproduce

    if opts.command == 'fit':
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # opts.default_root_dir = os.path.join(opts.result_root, opts.exp_name+'_'+now)
        opts.default_root_dir = os.path.join(opts.result_root, opts.exp_name)
        if not os.path.exists(opts.default_root_dir):
            os.makedirs(opts.default_root_dir)
            # code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
            # shutil.copytree(code_dir, os.path.join(opts.default_root_dir, 'code'))
            print('save in', opts.default_root_dir)
        else:
            print("Warning result_dir exists: (reload checkpoint)" + opts.default_root_dir)
            # sys.exit("result_dir exists: "+opts.default_root_dir)
        # solve learning rate
        bs, base_lr =  opts.batch_size, opts.base_learning_rate
        if opts.accelerator == 'cpu':
            ngpu = 1
        else:
            ngpu = len(opts.devices)
        if hasattr(opts, 'accumulate_grad_batches'):
            accumulate_grad_batches = opts.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        if opts.scale_lr:
            opts.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    opts.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            opts.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {opts.learning_rate:.2e}")