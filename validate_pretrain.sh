set -x

export HF_ENDPOINT="https://hf-mirror.com"

project_name=DeepScaleR-grpo
run_name=validate-pretrain

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

train_path=/data/personal/datasets/DeepScaleR-Preview/train.parquet
test_path=/data/personal/datasets/retool_aime2024/train.parquet

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo --config-path=config \
 --config-name='ppo_megatron_trainer.yaml'\
 trainer.val_before_train=True \
 +trainer.val_only=True \
 actor_rollout_ref.rollout.val_kwargs.n=1\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.trace.backend=null \
        actor_rollout_ref.rollout.trace.token2text=False \
        actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=null \
        actor_rollout_ref.rollout.multi_turn.enable=False \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
        actor_rollout_ref.rollout.multi_turn.interaction_config_path=null \
        actor_rollout_ref.rollout.agent.agent_loop_config_path=null \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.enable_activation_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1