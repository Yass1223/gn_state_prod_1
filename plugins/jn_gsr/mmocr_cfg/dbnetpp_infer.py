# dbnetpp_infer.py -- MMOCR 1.0.1 INFERENCE-ONLY config for the jersey-number
# DBNet++ detector.
#
# WHY THIS FILE EXISTS: the training config (jn_dbnetpp/mmocr_cfg/dbnetpp_jn.py)
# reads JN_SCRATCH / JN_WORK, globs an ICDAR init checkpoint, and points its
# datasets at converted Kaggle JSONs. None of that exists on a Lightning Studio,
# and none of it is needed to run a forward pass. This file carries ONLY what
# inference needs.
#
# The `model` dict below is copied BYTE-FOR-BYTE from the training config. A
# mismatch would load best_icdar_hmean_epoch_10.pth into a differently-shaped
# network and produce plausible-but-degraded detections rather than an error, so
# verify_pipeline.py asserts the two dicts are exactly equal at runtime.
#
# The ONE intentional deviation is the test pipeline: the training config's
# version includes LoadOCRAnnotations (it scores against GT). At inference there
# is no GT, so annotation loading is dropped. The Resize scale (736, keep_ratio)
# is kept identical -- that is the geometry the checkpoint was validated at.
_base_ = []  # sentinel: forces mmengine old-style (exec) parsing, same as the
             # training config -- without it the imports below make _is_lazy_import()
             # mis-parse this as a "new-style" config.

default_scope = "mmocr"

model = dict(
    type="DBNet",
    backbone=dict(
        type="mmdet.ResNet", depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=-1, norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False, style="pytorch",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50")),
    neck=dict(type="FPNC", in_channels=[256, 512, 1024, 2048],
              lateral_channels=256,
              asf_cfg=dict(attention_type="ScaleChannelSpatial")),
    det_head=dict(
        type="DBHead", in_channels=256,
        module_loss=dict(type="DBModuleLoss", shrink_ratio=0.4,
                         thr_min=0.3, thr_max=0.7),
        postprocessor=dict(type="DBPostprocessor", text_repr_type="quad",
                           unclip_ratio=1.5)),
    data_preprocessor=dict(
        type="TextDetDataPreprocessor",
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_size_divisor=32))

test_pipeline = [
    dict(type="LoadImageFromFile", color_type="color_ignore_orientation"),
    dict(type="Resize", scale=(736, 736), keep_ratio=True),
    dict(type="PackTextDetInputs",
         meta_keys=("img_path", "ori_shape", "img_shape", "scale_factor")),
]

# TextDetInferencer reads the pipeline from here and swaps the file loader for
# its own InferencerLoader so it can accept in-memory arrays as well as paths.
test_dataloader = dict(
    batch_size=1, num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(type="OCRDataset", data_root="", ann_file="",
                 test_mode=True, pipeline=test_pipeline))
val_dataloader = test_dataloader

test_cfg = dict(type="TestLoop")
val_cfg = dict(type="ValLoop")
test_evaluator = dict(type="HmeanIOUMetric")
val_evaluator = test_evaluator

env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
               dist_cfg=dict(backend="nccl"))
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="TextDetLocalVisualizer", name="visualizer",
                  vis_backends=vis_backends)
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffer=dict(type="SyncBuffersHook"),
    visualization=dict(type="VisualizationHook", enable=False))
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=20260713)
