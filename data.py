transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset_full = datasets.MNIST('../data/mnist', train=True, download=True, transform=transform_train)
test_dataset_full = datasets.MNIST('../data/mnist', train=False, download=True, transform=transform_test)

X_train_full = train_dataset_full.data.unsqueeze(1).float() / 255.0
y_train_full = train_dataset_full.targets

X_test_full = test_dataset_full.data.unsqueeze(1).float() / 255.0
y_test_full = test_dataset_full.targets


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_dataset_full = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=transform_train)
test_dataset_full = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=transform_test)

X_train_full = torch.tensor(train_dataset_full.data).permute(0, 3, 1, 2).float() / 255.0
y_train_full = torch.tensor(train_dataset_full.targets)
X_test_full = torch.tensor(test_dataset_full.data).permute(0, 3, 1, 2).float() / 255.0
y_test_full = torch.tensor(test_dataset_full.targets)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


train_dataset_full = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=transform_train)
test_dataset_full = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=transform_test)

X_train_full = torch.tensor(train_dataset_full.data).permute(0, 3, 1, 2).float() / 255.0
y_train_full = torch.tensor(train_dataset_full.targets)
X_test_full = torch.tensor(test_dataset_full.data).permute(0, 3, 1, 2).float() / 255.0
y_test_full = torch.tensor(test_dataset_full.targets)



def split_iid(train_targets, num_users, seed=42, min_size=10):
    rng = np.random.default_rng(seed)
    n = len(train_targets)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    base = n // num_users
    rem  = n % num_users

    dict_users = {}
    start = 0
    for i in range(num_users):
        size_i = base + (1 if i < rem else 0)
        part = idxs[start:start + size_i]
        start += size_i
        if len(part) < min_size:
            raise ValueError(f"Client {i} has only {len(part)} samples (< min_size={min_size}).")
        dict_users[i] = set(part.tolist())

    return dict_users


dict_users_train = split_iid(y_train_full, args.num_users, seed=42, min_size=10)


