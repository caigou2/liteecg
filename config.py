# config.py

# Data path configuration
DATA_ROOT = './data/mitbih'
DATA_PTB_ROOT = './data/ptb'
ECGNET_CONFIG = {
    'struct': [15, 17, 19, 21],
    'in_channels': 8,
    'fixed_kernel_size': 17,
    'num_classes': 5,
}

SE_ECGNET_CONFIG = {
    'struct': [(1, 3), (1, 5), (1, 7)],
    'num_classes': 4,
}

BIRCNN_CONFIG = {
    'num_classes': 4,
}

RESNET_CONFIG = {
    'num_classes': 4,
}

LITEECGNET_CONFIG = {
    'num_classes': 5,
    'fs': 360,
    'segment_len': 720,
    'base_channels':32
}

DEEPECGNET_CONFIG = {
    'in_ch': 1,
    'base_ch': 64,
    'd_model': 128,
    'n_transformer_layers': 2,
    'nhead': 4,
    'dim_ff': 256,
    'dropout': 0.1,
    'num_classes': 4,
    'hidden':128,
    'layers':2,
    'heads':4,
    'ff':256
}

LDCNN_CONFIG = {
    'num_classes': 5,
    'input_len': 360,
}

IMBECGNET_CONFIG = {
    'num_classes': 4,
    'fs': 360,
    'segment_len': 720,
    'attn_heads': 4,
}

IMBECGNET_IMPROVED_CONFIG = {
    'num_classes': 4,
    'fs': 360,
    'segment_len': 720,
    'attn_heads': 4,
}

IMBECGNET_ENHANCED_CONFIG = {
    'num_classes': 4,
    'fs': 360,
    'segment_len': 720,
    'attn_heads': 8,
}

IMBECGNET_CGHC_CONFIG = {
    'num_classes': 4,
    'fs': 360,
    'segment_len': 720,
    'attn_heads': 4,
}

# # LiteECGNet variant configuration examples
# NEW_ECGNET_LARGE_CONFIG = {
#     'num_classes': 4,
#     'fs': 360,
#     'segment_len': 720,
#     'attn_heads': 8,
#     'sinc_out_channels': 16,
#     'sinc_kernel_size': 41,
#     'conv7_out_channels': 32,
#     'conv7_kernel_size': 9,
#     'pw32_out_channels': 64,
#     'stage1_expand': 4,
#     'stage1_kernel': 7,
#     'stage2_expand': 4,
#     'stage2_kernel': 7,
#     'stage3_expand': 4,
#     'stage3_kernel': 7,
#     'tag_kernel': 13,
#     'dropout_rate': 0.2
# }

# NEW_ECGNET_SMALL_CONFIG = {
#     'num_classes': 4,
#     'fs': 360,
#     'segment_len': 720,
#     'attn_heads': 2,
#     'sinc_out_channels': 4,
#     'sinc_kernel_size': 21,
#     'conv7_out_channels': 8,
#     'conv7_kernel_size': 5,
#     'pw32_out_channels': 16,
#     'stage1_expand': 1,
#     'stage1_kernel': 3,
#     'stage2_expand': 1,
#     'stage2_kernel': 3,
#     'stage3_expand': 1,
#     'stage3_kernel': 3,
#     'tag_kernel': 5,
#     'dropout_rate': 0.05
# }

