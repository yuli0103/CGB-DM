from module.model import My_Model

# transformer_encoder = nn.TransformerEncoder(  # type: ignore
#             encoder_layer=nn.TransformerEncoderLayer(
#                 d_model=256,
#                 nhead=8,
#                 batch_first=True,
#                 dropout=0.1,
#                 norm_first=True,
#                 dim_feedforward=1024,
#             ),
#             num_layers=6,
#         )
model = My_Model(device='cuda:4')
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num
#
total_num, trainable_num = get_parameter_number(model)
print(f"trainable_num/total_num: %.2fM/%.2fM" % (trainable_num / 1e6, total_num / 1e6))

# device = torch.device(f"cuda:{6}" if torch.cuda.is_available() else "cpu")
# # 加载预训练的BERT模型和分词器
# model_profile = My_Model(device=device)
# mean = 0.0
# std = 1.0
# layout = torch.normal(mean, std, size=(32, 16, 8)).to(device)
# image = torch.normal(mean, std, size=(32, 4, 256, 256)).to(device)
# detect_box = torch.normal(mean, std, size=(32, 1, 4)).to(device)
# timestep = torch.normal(mean, std, size=(32,)).to(device)
# timestep = timestep.long()
# flops, params = profile(model_profile, inputs=(layout,image,detect_box,timestep), verbose=False)
#
# print(f"FLOPs: {flops}")
# print(f"Parameters: %.2fM" % (params / 1e6))