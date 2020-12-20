import cv2
import numpy
import torch
import torchvision
import torch.nn as nn
from tensorboardX import SummaryWriter


def test_scalar():
    ns = [2, 3, 4]
    for n in ns:
        writer = SummaryWriter(f'runs/scalar_example_{n:d}')
        for i in range(10):
            writer.add_scalar('quadratic', i**n, global_step=i)
            writer.add_scalar('exponential', n**i, global_step=i)
    writer.close()


def test_image():
    writer = SummaryWriter('runs/image_example')
    for i in range(0, 10):
        image = cv2.cvtColor(cv2.imread(f'images/{i}.png'), cv2.COLOR_BGR2RGB)
        writer.add_image('countdown', image, global_step=i, dataformats='HWC')
    writer.close()


def test_histogram():
    writer = SummaryWriter('runs/histogram_example')
    writer.add_histogram('normal_centered', numpy.random.normal(0, 1, 1000), global_step=1)
    writer.add_histogram('normal_centered', numpy.random.normal(0, 2, 1000), global_step=50)
    writer.add_histogram('normal_centered', numpy.random.normal(0, 3, 1000), global_step=100)
    writer.close()


def test_graph():
    writer = SummaryWriter('runs/graph_example')
    dummy_input = (torch.zeros(1, 3),)
    writer.add_graph(LinearInLinear(), dummy_input, True)
    writer.close()


class LinearInLinear(nn.Module):
    def __init__(self):
        super(LinearInLinear, self).__init__()
        self.l = nn.Linear(3, 5)

    def forward(self, x):
        return self.l(x)


def test_embedding():
    writer = SummaryWriter('runs/embedding_example')
    mnist = torchvision.datasets.MNIST('../dataset', download=True)

    data = mnist.train_data.reshape((-1, 28 * 28))[:100, :]
    metadata = mnist.train_labels[:100]
    label_img= mnist.train_data[:100,:,:].reshape((-1, 1, 28, 28)).float() / 255

    writer.add_embedding(
        data,
        metadata=metadata,
        label_img = label_img,
        global_step=0
    )
    writer.close()

if __name__ == "__main__":
    test_scalar()
    test_image()
    test_histogram()
    test_graph()
    test_embedding()
