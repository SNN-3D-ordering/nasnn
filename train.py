import torch
import torch.nn as nn
from utils import train_printer


def train_model(
    net, train_loader, test_loader, device, num_steps, num_epochs, batch_size, dtype
):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    loss_hist = []
    test_loss_hist = []
    counter = 0

    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))

            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())

            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                test_spk, test_mem = net(test_data.view(batch_size, -1))

                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                if counter % 50 == 0:
                    train_printer(
                        epoch,
                        iter_counter,
                        loss_hist,
                        test_loss_hist,
                        data,
                        targets,
                        test_data,
                        test_targets,
                        net,
                        batch_size,
                    )
                counter += 1
                iter_counter += 1

    return loss_hist, test_loss_hist
