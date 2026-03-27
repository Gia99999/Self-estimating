@torch.no_grad()
def eval_acc(net, X_eval, y_eval):
    net.eval()
    loader = DataLoader(TensorDataset(X_eval, y_eval), batch_size=args.bs, shuffle=False)
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(args.device), yb.to(args.device)
        pred = net(xb).argmax(dim=1)
        correct += (pred == yb).long().sum().item()
        total += yb.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def test_img(net_g, data_loader):
    net_g.eval()
    correct = 0
    for data, target in data_loader:
        data, target = data.to(args.device), target.to(args.device)
        out = net_g(data)
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).long().cpu().sum().item()
    return 100.0 * correct / len(data_loader.dataset)
  
class LocalUpdate(object):
    def __init__(self, args, noisy_data):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.noisy_data = noisy_data
        self.ldr_train = self.get_noisy_loader()

    def get_noisy_loader(self):
        images, labels = self.noisy_data
        ds = TensorDataset(images, labels)
        return DataLoader(ds, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        opt = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        last_loss = None
        for _ in range(self.args.local_ep):
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                opt.zero_grad()
                out = net(images)
                loss = self.loss_func(out, labels)
                loss.backward()
                opt.step()
                last_loss = loss.item()
        return copy.deepcopy(net.state_dict()), last_loss
      
  def FedAvg(w_locals, lens=None, weights=None, eps=1e-12):
    n = len(w_locals)
    assert n > 0, "Empty"
    if (lens is None) == (weights is None):
        raise ValueError("Provide weights")

    used = np.asarray(lens if lens is not None else weights, dtype=np.float64)
    used = np.clip(used, 0.0, None)
    s = used.sum()
    if s <= 0:
        used = np.ones(n, dtype=np.float64)
        s = used.sum()
    used = used / (s + eps)

    w_avg = {}
    for k in w_locals[0].keys():
        t0 = w_locals[0][k]
        if not torch.is_floating_point(t0):
            w_avg[k] = t0.clone()
            continue
        acc = torch.zeros_like(t0)
        for i in range(n):
            acc += float(used[i]) * w_locals[i][k]
        w_avg[k] = acc

    return w_avg, used

all_runs_results = {name: [] for name in variants.keys()}
for run in range(num_runs):
    noisy_train_data = {}
    noisy_val_data   = {}

    for idx in range(args.num_users):
        train_idxs = list(dict_users_train[idx])
        X_client = X_train_full[train_idxs]
        y_client = y_train_full[train_idxs]

        X_tr, X_val, y_tr, y_val = safe_train_val_split(X_client, y_client, val_ratio, seed=idx)

        noise_ratio = float(client_noises[idx])
        y_tr_noisy  = add_noise(y_tr,  noise_ratio, seed=1000 + idx)
        y_val_noisy = add_noise(y_val, noise_ratio, seed=2000 + idx)

        noisy_train_data[idx] = (X_tr, y_tr_noisy)
        noisy_val_data[idx]   = (X_val, y_val_noisy)
      
    client_slopes = np.zeros(args.num_users, dtype=float)

    for idx in range(args.num_users):
        X_tr, y_tr_noisy = noisy_train_data[idx]
        X_val, y_val_noisy = noisy_val_data[idx]

        slope = compute_client_slope_1_reinit(
            X_tr, y_tr_noisy,
            X_val, y_val_noisy
        )
        client_slopes[idx] = slope

    for variant_name, mode in variants.items():
    
        net_glob = build_model()
        acc_list = []

        for round_num in range(args.epochs)
            m = max(int(args.frac * args.num_users), 1)
            selected_clients = np.random.choice(range(args.num_users), m, replace=False)
            w_locals_round = []
            selected_slopes = []
            selected_lens = []

            for idx in selected_clients:
                local = LocalUpdate(args=args, noisy_data=noisy_train_data[idx])
                w, _ = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals_round.append(w)

                selected_slopes.append(client_slopes[idx])
                selected_lens.append(len(noisy_train_data[idx][1])) 

            if mode is None:
                w_glob_new, _ = FedAvg(w_locals_round, lens=selected_lens, weights=None)
            else:
                weights = slopes_to_weights(selected_slopes, temp=1.0)
                w_glob_new, _ = FedAvg(w_locals_round, lens=None, weights=weights)

            net_glob.load_state_dict(w_glob_new)

            acc = test_img(net_glob, test_loader)
            acc_list.append(float(acc))

        all_runs_results[variant_name].append(acc_list)
