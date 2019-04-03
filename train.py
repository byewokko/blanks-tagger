import sys
import torch
import torch.nn as nn
import torch.optim as optim

import util

from util import stderr_print
from confusion_matrix import ConfusionMatrix
from timer import Timer, timestamp


class CustomDropout(nn.Module):
    def __init__(self, unk_value=2, p=0.5):
        super(CustomDropout, self).__init__()
        self.unk_value = unk_value
        self.p = p

    def forward(self, X):
        Y = X.bernoulli(p=self.p) * self.unk_value
        Y[Y == 0] = X[Y == 0]
        return Y


class Net(nn.Module):
    def __init__(self, vocab_size, tagset_size, pad_value, unk_value, **kwargs):
        super(Net, self).__init__()
        self.unk_value = unk_value
        self.pad_value = pad_value
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.embedding_dim = 20
        self.rnn_layer_size = 80
        self.rnn_layer_number = 3
        self.bidirectional = True
        #self.cell_type = "RNN_TANH"
        self.cell_type = "LSTM"
        self.dropout_rate = 0.2
        self.tagset_size = tagset_size

        self.rnn_out_size = self.rnn_layer_size * (self.bidirectional + 1)

        self.__build_model()

    def __build_model(self):
        self.unk_layer = CustomDropout(unk_value=self.unk_value,
                                       p=0.002)
        self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=self.pad_value)
        self.rnn_layer = nn.RNNBase(mode=self.cell_type,
                                    input_size=self.embedding_dim,
                                    hidden_size=self.rnn_layer_size,
                                    num_layers=self.rnn_layer_number,
                                    bidirectional=self.bidirectional,
                                    dropout=self.dropout_rate,
                                    batch_first=True)
        self.hidden_layer = None
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.dense_layer = nn.Linear(self.rnn_out_size, self.tagset_size)
        self.activation_layer = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        rnn_layer_number = self.rnn_layer_number * (self.bidirectional + 1)
        if self.cell_type == "LSTM":
            hidden = torch.randn(rnn_layer_number, batch_size, self.rnn_layer_size)
            self.hidden_layer = hidden, hidden
        else:
            hidden = torch.randn(rnn_layer_number, batch_size, self.rnn_layer_size)
            self.hidden_layer = hidden

    def forward(self, X, L):
        seq_length = X.shape[1]
        if self.training:
            X = self.unk_layer(X)
        X = self.emb_layer(X)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, L, batch_first=True)
        X, self.hidden_layer = self.rnn_layer(X, self.hidden_layer)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = X.contiguous().view(-1, X.shape[2])
        X = self.dropout_layer(X)
        X = self.dense_layer(X)
        X = self.activation_layer(X)
        X = X.view(-1, seq_length, self.tagset_size)
        Yh = X
        return Yh

    def save(self, filename):
        torch.save(self, filename)
        stderr_print(f"Model saved to {filename}")

    def save_embeddings(self, i2char, filename):
        with open(filename, "w") as fout:
            for i, c in i2char.items():
                vector = [c]
                for n in self.emb_layer.weight[i]:
                    vector.append(f"{n:.8f}")
                print("\t".join(vector), file=fout)
        stderr_print(f"Embeddings saved to {filename}")


def predict(model, dataloader, loss_function, conf_matrix, n_batches=None):
    n_batches = n_batches or len(dataloader)
    epoch_loss = 0

    for batch_n, (X, Y, L) in enumerate(dataloader):
        if batch_n == n_batches:
            break
        stderr_print("Evaluating |{}|".format(util.loadbar(batch_n / (n_batches - 1))), end="\r")

        model.zero_grad()
        model.hidden = model.init_hidden(batch_size=len(L))

        Yh = model(X, L)

        Yh_tags = Yh.max(dim=2)[1]
        conf_matrix.add(Yh_tags, Y)

        with torch.no_grad():
            loss = loss_function(Yh.view(-1, Yh.shape[2]), Y.view(-1))
        epoch_loss += loss

    return epoch_loss


def train_epoch(model, dataloader, loss_function, optimizer, conf_matrix):
    n_batches = len(dataloader)
    epoch_loss = 0
    timer = Timer(total_ticks=n_batches)
    timer.start()

    for batch_n, (X, Y, L) in enumerate(dataloader):
        stderr_print("Training |{}| {}".format(util.loadbar(batch_n / (n_batches - 1)), timer.remaining()), end="\r")
        model.zero_grad()
        model.hidden = model.init_hidden(batch_size=len(L))

        Yh = model(X, L)

        Yh_tags = Yh.max(dim=2)[1]
        # conf_matrix.add(Yh_tags, Y)
        # print(Yh.shape, Y.shape)

        loss = loss_function(Yh.view(-1, Yh.shape[2]), Y.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss

        timer.tick()

    return epoch_loss


def train(model, train_loader, dev_loader, n_epochs, confmat, out_dir, eval_batches=None):
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    eval_batches = min(eval_batches or len(train_loader), len(train_loader), len(dev_loader))
    train_confmat = confmat
    dev_confmat = train_confmat.copy()
    log = [{}]
    best_dev_loss = (None, 0)

    for e in range(1, n_epochs + 1):
        log.append({})
        model = model.train()
        train_loss = train_epoch(model, train_loader,
                                 loss_function=loss_function,
                                 optimizer=optimizer,
                                 conf_matrix=train_confmat)

        model = model.eval()
        train_confmat.reset()
        train_loss = predict(model, dev_loader,
                             loss_function=loss_function,
                             conf_matrix=train_confmat,
                             n_batches=eval_batches)

        log[e]["train_loss"] = train_loss / eval_batches
        log[e]["train_acc"] = train_confmat.accuracy()
        log[e]["train_f1"] = train_confmat.f_score(mean=True)
        print(f"[{timestamp()}] Epoch {e}: T "
              f"loss {log[e]['train_loss']:.6f}, "
              f"accuracy {log[e]['train_acc']:.4f}, "
              f"f1 {log[e]['train_f1']:.4f}")

        dev_confmat.reset()
        dev_loss = predict(model, dev_loader,
                           loss_function=loss_function,
                           conf_matrix=dev_confmat,
                           n_batches=eval_batches)

        log[e]["dev_loss"] = dev_loss / eval_batches
        log[e]["dev_acc"] = dev_confmat.accuracy()
        log[e]["dev_f1"] = dev_confmat.f_score(mean=True)
        print(f"[{timestamp()}] Epoch {e}: E "
              f"loss {log[e]['dev_loss']:.6f}, "
              f"accuracy {log[e]['dev_acc']:.4f}, "
              f"f1 {log[e]['dev_f1']:.4f}")

        if best_dev_loss[0] is None or log[e]["dev_loss"] < best_dev_loss[0]:
            best_dev_loss = log[e]["dev_loss"], e
            model.save(out_dir + "-check.model")

    if log[-1]["dev_loss"] > best_dev_loss[0]:
        print(f"Loading best model (epoch {best_dev_loss[1]})...")
        model = torch.load(out_dir + "-check.model")

    return model


def main():
    print(f"[{timestamp('%Y-%m-%d %H:%M:%S')}] train.py")
    remove_characters = "aeiouy"

    # data_dir = "data"
    data_dir = "/home/m17/hruska/nobackup/cs/deacc"

    train_file = data_dir + "/train-mini.txt"
    #train_file = data_dir + "/train.txt"
    dev_file = data_dir + "/dev.txt"
    test_file = data_dir + "/test.txt"

    #out_dir = "/home/m17/hruska/nobackup/out"
    if len(sys.argv) > 1:
        out_prefix = sys.argv[1]
    else:
        out_prefix = "/home/m17/hruska/nobackup/out/xxx"

    batch_size = 128

    stderr_print("Preparing data...")
    inchar2i, i2inchar, outchar2i, i2outchar = util.make_index_dicts(train_file, remove_characters)
    train_loader, dev_loader, test_loader = util.prepare_dataloaders(train_file, dev_file, test_file,
                                                                     inchar2i, outchar2i, batch_size)

    stderr_print("Setting up model...")
    model = Net(vocab_size=len(inchar2i),
                tagset_size=len(outchar2i),
                pad_value=0,
                unk_value=inchar2i["<UNK>"])
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    confmat = ConfusionMatrix(len(outchar2i),
                              ignore_index=outchar2i["<PAD>"],
                              class_dict=i2outchar)

    stderr_print("Training...")
    model = train(model, train_loader, dev_loader,
                  n_epochs=20,
                  confmat=confmat,
                  out_dir=out_prefix)

    stderr_print("Saving model...")
    torch.save(model, out_prefix + "-model.model")
    stderr_print("Saving embeddings...")
    model.save_embeddings(i2inchar, out_prefix + "-embs.txt")


if __name__ == "__main__":
    main()
