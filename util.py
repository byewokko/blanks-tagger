import sys
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader


def sort_batch(X, Y, L, descending=True):
    """
    Sort batch according to length L.
    :param X:
    :param Y:
    :param L:
    :return:
    """
    L_sorted, idx_sorted = L.sort(0, descending=descending)
    X_sorted = X[idx_sorted]
    Y_sorted = Y[idx_sorted]

    return X_sorted, Y_sorted, L_sorted


def pad_sort_batch(batch, padding_value=0):
    """
    Collate function that pads the current batch.
    :param batch:
    :param pad_value:
    :return:
    """
    # batch_item = (x, y, length)
    X = []
    Y = []
    L = []
    max_length = max(map(lambda x: x[2], batch))

    for x, y, l in batch:
        X.append(F.pad(x,
                       pad=(0, (max_length - l)),
                       mode="constant",
                       value=padding_value))
        Y.append(F.pad(y,
                       pad=(0, (max_length - l)),
                       mode="constant",
                       value=padding_value))
        L.append(l)

    X = torch.stack(X)
    Y = torch.stack(Y)
    L = torch.LongTensor(L)

    batch_sorted = sort_batch(X, Y, L)

    return batch_sorted


def make_index_dicts(filename, remove_characters, lower=True):
    outchar2i = {"<PAD>": 0}
    for c in remove_characters:
        outchar2i[c] = len(outchar2i)

    inchar2i = {"<PAD>": 0, "<XXX>": 1, "<UNK>": 2}
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            if lower:
                line = line.lower()
            for c in line:
                if c not in outchar2i and c not in inchar2i:
                    inchar2i[c] = len(inchar2i)

    i2inchar = {v: k for k, v in inchar2i.items()}
    i2outchar = {v: k for k, v in outchar2i.items()}

    return inchar2i, i2inchar, outchar2i, i2outchar


def index_data(filename, inchar2i, outchar2i, lower=True):
    data = []

    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            if lower:
                line = line.lower()
            x = []
            y = []
            for c in line:
                if c in outchar2i:
                    x.append(inchar2i["<XXX>"])
                    y.append(outchar2i[c])
                elif c in inchar2i:
                    x.append(inchar2i[c])
                    y.append(outchar2i["<PAD>"])
                else:
                    x.append(inchar2i["<UNK>"])
                    y.append(outchar2i["<PAD>"])

            assert len(x) == len(y)
            data.append((torch.LongTensor(x), torch.LongTensor(y), len(x)))

    #X = torch.stack(X)
    #Y = torch.stack(Y)
    #L = torch.LongTensor(L)
    return data


def prepare_dataloaders(train_file, dev_file, test_file,
                        inchar2i, outchar2i,
                        batch_size, collate_fn=pad_sort_batch, num_workers=8):

    train = index_data(train_file, inchar2i, outchar2i)
    dev = index_data(dev_file, inchar2i, outchar2i)
    test = index_data(test_file, inchar2i, outchar2i)

    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dev,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


def loadbar(percent, n_blocks=20):
    percent = min(percent, 0.999999999)
    blocks = [b for b in "▏▎▍▌▋▊▉█"]
    whole = percent * n_blocks
    part = (whole - int(whole)) * len(blocks)
    return int(whole)*"█" + blocks[int(part)] + int(n_blocks - int(whole) - 1)*"-"


def stderr_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()
