from scratch import dataset

# %% 初版实现
def get_data():
    (x_train, t_train), (x_test, t_test) = dataset.mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
    
def init_network():
    with open("")

if __name__ == "__main__":
    print("haha")
