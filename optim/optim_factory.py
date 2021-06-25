import torch.optim as optim


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def build_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.wd
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        params = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        # If distillation mode, only pass in student parameters + temperature.
        params = filter(lambda p: p.requires_grad, model.parameters())

    if opt_lower == "sgd":
        optimizer = optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError

    return optimizer