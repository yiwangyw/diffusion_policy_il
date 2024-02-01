#!/bin/bash
# PYTHON=`which python3`
for i in {1..100}
do
#    PYTHON eval_diffusion.py
   python diffusion_policy/workspace/eval_diffusion.py --config-dir=./configs/image/square_ph/diffusion_policy_cnn --config-name=config.yaml hydra.run.dir=data/eval/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_square_ph_ddp_eval_diffusion logging.project=eval-test logging.name=$(date +%Y.%m.%d)_$(date +%H.%M.%S)_square_ph_ddp_eval_diffusion
done