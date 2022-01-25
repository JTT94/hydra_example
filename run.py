
import torch
import hydra
from hydra.utils import instantiate, get_class, call
import omegaconf
from src.loss import loss_function
from torchvision.utils import save_image


def run(cfg: omegaconf):
    device = cfg.device.device
    dataset = instantiate(cfg.dataset)

    optimizer = instantiate(cfg.optim)
    model = instantiate(cfg.model).to(device)

    kwargs = {'num_workers': cfg.device.num_workers, 'pin_memory': cfg.device.pin_memory} 

    train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=cfg.training.batch_size, shuffle=True, **kwargs)





    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))


    for epoch in range(1, cfg.training.num_epochs + 1):
        train(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(cfg,device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

    
    
