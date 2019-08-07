import torch
import unittest

from recurrent_models import StackedQLSTM

class TestStackedQLSTM(unittest.TestCase):
    feat_size = 4 # Must be multiple of 4
    batch_size = 2
    seq_len = 5
    hidden_size = 16 # Must be multiple of 4

    def test_train(self):
        lstm = StackedQLSTM(self.feat_size, self.hidden_size, False, 2, True)
        inputs = torch.ones((self.batch_size, self.seq_len, self.feat_size))

        optimizer = torch.optim.SGD(lstm.parameters(), lr=0.001)
        lstm.train()

        init_loss = None

        for i in range(50):
            optimizer.zero_grad()
            outputs = lstm(inputs)
            loss = torch.nn.L1Loss()(outputs, torch.ones(outputs.size()))
            if init_loss is None:
                init_loss = loss.item()
            loss.backward()
            optimizer.step()
            print("Loss: {}".format(loss.item()))

        self.assertLess(loss.item(), init_loss)

if __name__ == "__main__":
    unittest.main()
