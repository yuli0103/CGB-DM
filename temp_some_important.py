
# 使用新的方式加载预训练的 ResNet50 模型
weights = ResNet50_Weights.IMAGENET1K_V1  # 或者使用 ResNet50_Weights.DEFAULT
self.img_encoder = models.resnet50(weights=weights)
# self.img_encoder = models.resnet50(pretrained=True)
original_first_layer = self.img_encoder.conv1.weight
# 创建新的卷积层，输入通道为 4
new_first_layer = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# 复制原始层的权重到新层的前三个通道
with torch.no_grad():
    new_first_layer.weight[:, :3] = original_first_layer
    # 对第四个通道的权重进行初始化（这里我选择复制第一个通道的权重）
    new_first_layer.weight[:, 3] = original_first_layer[:, 0]
# 用新的卷积层替换原始模型的第一个卷积层
self.img_encoder.conv1 = new_first_layer
self.img_encoder = torch.nn.Sequential(*(list(self.img_encoder.children())[:-1])).to(self.device)
self.img_encoder.eval()

# self.img_encoder = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], max_elem=max_elem, d_model=d_model).to(self.device)
model_name = "google/vit-base-patch16-224-in21k"
# self.img_encoder = ViTImageProcessor.from_pretrained(model_name, image_channel_format="channels_first")
# self.img_encoder = ViTForImageClassification.from_pretrained(model_name)

terms["mse"] = mean_flat(th.abs(target - model_output))
print(terms["mse"])
sys.exit()
coef = [0.1, 0.8, 1, 1]
empty_weight = th.tensor(coef).to(device)
x_cls = x_cls.view(-1, 4)
x_cls = nn.Softmax(dim=1)(x_cls)
output_cls = output_cls.view(-1, 4)
loss_ce = F.cross_entropy(x_cls.float(), output_cls.float(), empty_weight)
terms["cls"] = loss_ce
print(terms["cls"])
sys.exit()
target = target.view(-1, 4)
output_box = output_box.view(-1, 4)
loss_giou = 1 - th.diag(generalized_box_iou(
    box_cxcywh_to_xyxy(target),
    box_cxcywh_to_xyxy(output_box)))
terms["giou"] = loss_giou

pbar.set_description(f'Epoch {epoch + 1} / Epochs {self.total_epochs}, '
                     f'LR: {(self.opt.param_groups[0]["lr"]):.2e}, '
                     f'AVG_Loss_bbox: {total_loss_bbox / train_steps:.4f}, '
                     f'AVG_Loss_cls: {total_loss_ce / train_steps:.4f}, '
                     f'AVG_Loss_giou: {total_loss_giou / train_steps:.4f}')











def run_step(self, image, label, detect_box, epoch):
    self.opt.zero_grad()
    # self.opt_enc.zero_grad()

    t = self.model_ddpm.sample_t([label.shape[0]], t_max=self.model_ddpm.num_timesteps - 1)
    label = label.reshape(label.shape[0], label.shape[1], -1)
    label[:, :, 4:] = 2 * (label[:, :, 4:] - 0.5)

    detect_box = detect_box.reshape(detect_box.shape[0], detect_box.shape[1], -1)
    detect_box[:, :, 4:] = 2 * (detect_box[:, :, 4:] - 0.5)

    eps_theta, e, b_0_reparam = self.model_ddpm.forward_t(label, image, detect_box,
                                                          t=t, real_mask=None, reparam=True)
    # 计算损失
    _, target_classes = b_0_reparam[:, :, :4].max(dim=2)  # 形状为(batch_size, num_boxes)
    # target_classe = torch.argmax(b_0_reparam[:, :, :4], dim=2)
    mask = (target_classes == 1) | (target_classes == 2)  # 形状为(batch_size, num_boxes)
    bbox_rep = torch.clamp(b_0_reparam[:, :, 4:], min=-1, max=1) / 2 + 0.5

    piou = PIoU_xywh(bbox_rep, mask=mask.to(torch.float32), xy_only=False)

    pdist = Pdist(bbox_rep)
    overlap_loss = torch.mean(piou, dim=[1, 2]) + torch.mean(piou.ne(0) * torch.exp(-pdist), dim=[1, 2])


    # layout_input_all = torch.cat([label, label, label], dim=0)
    mse_loss = nn.MSELoss()
    reconstruct_loss = mse_loss(label[:, :, 4:], b_0_reparam[:, :, 4:])

    # # local alignment
    align_mask = (target_classes != 0)
    _, align_loss = layout_alignment(bbox_rep, align_mask, xy_only=False)

    weight = constraint_temporal_weight(t, schedule='linear')
    align_loss = (align_loss * weight).mean()
    overlap_loss = (overlap_loss * weight).mean()
    # constraint_loss = torch.mean(align_loss + overlap_loss)

    diffusion_loss = mse_loss(e, eps_theta)

    # if epoch <250:
    #     loss = diffusion_loss
    # else:
    loss = diffusion_loss
    loss.backward()
    self.optimize_normal()
    return diffusion_loss, align_loss,  overlap_loss