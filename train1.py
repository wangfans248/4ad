from models.mymodel import mymodel

model = mymodel(config)
model.to(device)

for iteration, data_item in enumerate(input_data):
    model.dsc_opt.zero_grad()
    if model.pre_proj > 0:
        model.proj_opt.zero_grad()

    fake_features, true_features, noisy_features, r_t = model(data_item)

    # 计算损失并反向传播
    scores = model.discriminator(torch.cat([true_features, noisy_features]))
    # 进一步处理损失
    loss = compute_loss(scores)  # 具体的损失计算方法

    loss.backward()
    model.dsc_opt.step()  # 优化器步骤
    if model.pre_proj > 0:
        model.proj_opt.step()  # 如果使用预投影优化器，更新