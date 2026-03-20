set -x

export HF_ENDPOINT="https://hf-mirror.com"
export YOUR_PROJECT_NAME="gsm8k-grpo"
export YOUR_RUN_NAME="gsm8k-grpo"

# export NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

gsm8k_train_path=/data/personal/datasets/gsm8k/train.parquet
gsm8k_test_path=/data/personal/datasets/gsm8k/test.parquet

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo --config-path=config \
 --config-name='ppo_megatron_trainer.yaml'\
 algorithm.adv_estimator=grpo \
 data.train_files="$gsm8k_train_path" \
 data.val_files="$gsm8k_test_path" \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
 trainer.logger='["console","wandb"]' \
 trainer.project_name=$YOUR_PROJECT_NAME \
 trainer.experiment_name=$YOUR_RUN_NAME \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=50 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee $YOUR_RUN_NAME.log