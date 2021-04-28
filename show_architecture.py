from resnet_imagenet import resnet18
from resnet import resnet20

# model = resnet18()
# print(model)

model = resnet20().cpu()

import copy
import torch
fused_model = copy.deepcopy(model)

# print(fused_model)
# fuse the layers in the frontend
fused_model = torch.quantization.fuse_modules(fused_model,
                                                [["conv_1_3x3", "bn_1", "relu"]],
                                                inplace=True)

# print(fused_model)
for module_name, module in fused_model.named_children():
    if "stage" in module_name:
        for basic_block_name, basic_block in module.named_children():
            torch.quantization.fuse_modules(
                    basic_block, [["conv_a", "bn_a", "relu_a"], ["conv_b", "bn_b"]],
                    inplace=True)
            # for sub_block_name, sub_block in basic_block.named_children():
            #     if sub_block_name == "downsample":
            #         print(sub_block_name)

# print(fused_model)