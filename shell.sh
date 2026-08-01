#!/bin/bash

echo "started"

export $(grep -v '^#' .env | xargs)

echo "directory is $PROJECT_PATH"
echo "python directory read in"

source /etc/profile.d/modules.sh
module load cuda/12.9
echo "modules loaded"

cd "$PROJECT_PATH"
source "$POETRY_VENV/bin/activate"
echo "virtualenv activated"

export LAUNCH_MODE="batch"

torchrun --nproc_per_node=$NUM_GPUS -m src.poet.train
# torchrun --nproc_per_node=$NUM_GPUS -m src.poet.finetune_sae
# torchrun --nproc_per_node=1 -m src.poet.eval