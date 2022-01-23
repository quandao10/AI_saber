import torch
import torch.nn as nn
import torch.nn.functional as F
import neptune.new as neptune

run = neptune.init(
    project="quan-ml/AISABER",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDgzYmY3Zi1kMjZjLTRkNjUtYWY2Ny0wODAwZDBjNjkwNGUifQ==",
)  # your credentials here


def train(args, model, beat_dataloader, device, criterion, optimizer):
    params = vars(args)
    run["parameters"] = params
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    for epoch in range(args.epochs):
        for i, data in enumerate(beat_dataloader):
            map_input, map_target, audio_input = data
            map_input = map_input.float().to(device)
            audio_input = audio_input.float().to(device)
            map_target = map_target.long().to(device)

            optimizer.zero_grad()
            map_pred = model(map_input, audio_input)
            map_pred = map_pred[:, 0:args.map_target_length, :, :]
            map_pred = torch.permute(map_pred, (0, 3, 1, 2))
            loss = criterion(map_pred, map_target)

            loss.backward()
            optimizer.step()

            run["training/loss"].log(loss.item())

            if (i + 1) % args.log_interval == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, args.epochs, i + 1,
                                                                         len(beat_dataloader), loss.item()))
        lr_schedule.step(loss)
        torch.save(model.state_dict(), 'checkpoint/model_{}.pth'.format(epoch + 1))
