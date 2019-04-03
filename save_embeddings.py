import sys
import torch


def main():
    model = torch.load(sys.argv[1])
    model.save_embeddings(i2inchar, "embeddings.txt")


if __name__ == "__main__":
    main()



