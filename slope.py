def train_from_state(net, state_dict, X, y):
    net.load_state_dict(state_dict)
    net.train()
    opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    loader = DataLoader(TensorDataset(X, y), batch_size=args.local_bs, shuffle=True)
    for _ in range(args.local_ep):
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), yb)
            loss.backward()
            opt.step()
    return copy.deepcopy(net.state_dict())


def compute_client_slope(X_train, y_train_noisy, X_val, y_val_noisy,
                                change_grid=None, num_repeats=10):
    if change_grid is None:
        change_grid = np.arange(0, 1.1, 0.1)

    slopes = []
    for rep in range(num_repeats):
        rep_seed = seed + 10000 * rep
        set_global_seed(rep_seed)

        net0 = build_model()
        init_state = copy.deepcopy(net0.state_dict())
        base_state = train_from_state(net0, init_state, X_train, y_train_noisy)
        net_eval = build_model()
        net_eval.load_state_dict(base_state)
        acc1 = eval_acc(net_eval, X_val, y_val_noisy) + 1e-12

        rel = []
        for r in change_grid:
            y_changed = random_change_labels(y_train_noisy, r, seed=rep_seed + int(r * 1000))
            net_r = build_model()
            state_r = train_from_state(net_r, base_state, X_train, y_changed)
            net_r.load_state_dict(state_r)
            acc2 = eval_acc(net_r, X_val, y_val_noisy)
            rel.append((acc2 - acc1) / acc1)

        slope, _, _, _, _ = linregress(change_grid, rel)
        slopes.append(float(slope))

    return float(np.mean(slopes))


def slopes_to_weights(slopes, temp=1.0):
    s = np.asarray(slopes, dtype=float)
    lo, hi = np.quantile(s, clip_q[0]), np.quantile(s, clip_q[1])
    s = np.clip(s, lo, hi)

    z = (-s) / max(temp, eps)
    z = z - np.max(z)
    w = np.exp(z)
    w = w / (w.sum() + eps)
    return w
