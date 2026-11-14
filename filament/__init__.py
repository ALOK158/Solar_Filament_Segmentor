"""Solar filament instance segmentation for the IEEE / Kaggle Solar Filament Segmentation Challenge 2026.

The system, end to end:
    GONG H-alpha full-disk image (2048 x 2048, grayscale)
      -> three hand-designed input channels at 1536 px             (preprocess)
      -> three ConvNeXt-T (DINOv3) U-Nets, 4-flip TTA, averaged      (model, inference)
      -> filament-probability and spine-probability maps
      -> connected components = candidate filaments                 (decode)
      -> learned keep/drop filter on 15 shape/confidence features   (instance_filter)
      -> each kept filament re-segmented at native 2048 px          (refiner)
      -> pixel-disjoint COCO-RLE instances                          (submission)

Modules, in pipeline order:
    config           every setting, as frozen dataclasses with their provenance
    data             competition files, COCO annotations, annotator records, folds
    preprocess       grayscale image -> 3 input channels
    model            the U-Net; fp16 weight files on disk, fp32 in memory
    inference        TTA ensemble -> native-resolution probability maps
    decode           probability map -> instance masks
    instance_filter  per-instance features and the gradient-boosted keep/drop scorer
    refiner          stage 2: rotated native-resolution crop per kept filament, refinement, disjoint paste-back
    metrics          panoptic quality and the rubric's agreement statistics
    submission       RLE encoding, CSV writing, the host's format checks
    augment, datasets, losses, training, pseudo_labels   (training only)
    onnx_backend     ONNX export and ONNX Runtime inference
    maps             compressed uint8 storage of validation probability maps
    timing           per-stage wall-clock accounting
    viz              figures used by the notebook
    pipeline         end-to-end helpers the notebook calls
"""
__version__ = "1.0.0"
