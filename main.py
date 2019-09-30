from pre_process import pre_process_data
from model import train


def main():
    data, labels = pre_process_data('data', True)
    train(data, labels)


if __name__ == "__main__":
    main()
