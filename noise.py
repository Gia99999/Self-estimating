def add_noise(y, noise_ratio, seed):
    rng = np.random.default_rng(seed)
    y = y.clone()
    n = len(y)
    m = int(noise_ratio * n)
    if m <= 0:
        return y
    idx = rng.choice(n, m, replace=False)
    labels = y.unique().cpu().numpy()
    for i in idx:
        cur = int(y[i].item())
        y[i] = int(rng.choice(labels[labels != cur]))
    return y


def random_change_labels(y, change_ratio, seed):
    rng = np.random.default_rng(seed)
    y = y.clone()
    n = len(y)
    m = int(change_ratio * n)
    if m <= 0:
        return y
    idx = rng.choice(n, m, replace=False)
    labels = y.unique().cpu().numpy()
    for i in idx:
        cur = int(y[i].item())
        y[i] = int(rng.choice(labels[labels != cur]))
    return y
