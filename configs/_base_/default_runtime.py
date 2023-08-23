wandb_path = '/home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/tools/my_script/detr/train/wandb_results/detr'
wandb_project_name = 'LOCAL_detr_sparse'
wandb_run_name = 'reformer_epoch300_batch4_enc2_dec2'


default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=25),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


# vis_backends = [dict(type='LocalVisBackend')]


vis_backends = [dict(type='LocalVisBackend'), 
                dict( type='WandbVisBackend',
                      save_dir=wandb_path,
                      init_kwargs=dict(
                      project=wandb_project_name,
                      name=wandb_run_name,),
                      define_metric_cfg=None,
                      commit=True,
                      log_code_name=None,
                      watch_kwargs=None 
                       ) ]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
