import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from model import BodySeg
from dataset import HumanSeg
from torch.utils.tensorboard import SummaryWriter

train_dataset_path = '/path to/train_data'
eval_dataset_path = '/path to/eval_data'
checkpoints_path = '/path to/checkpoints/'
log_dir = '/path to/runs'  # location of tensorboard logs
body_parts = 4
batch_size = 1
lr = 0.0001
epoch = 500

transform = T.Compose([
    # T.RandomJitter(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2),
])
pre_transform = T.NormalizeScale()

train_dataset = dataset = HumanSeg(train_dataset_path, include_normals=False, transform=transform, body_part=body_parts)
eval_dataset = dataset = HumanSeg(eval_dataset_path, include_normals=False, transform=transform, body_part=body_parts)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BodySeg(3, train_dataset.num_classes, dim_model=[32, 64, 128, 256, 512], k=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
writer = SummaryWriter(log_dir=log_dir)  # tensorboard --logdir /runs

def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        data.y = data.y.squeeze(1)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes
        acc = correct_nodes / total_nodes

    # print(f'Loss: {total_loss / 10:.4f} 'f'Train Acc: {acc:.4f}')

    return acc, total_loss


def validation(loader):
    model.eval()

    correct_nodes = total_nodes = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pos, data.batch)
        correct_nodes += out.argmax(dim=1).eq(data.y.squeeze(1)).sum().item()
        total_nodes += data.num_nodes
    eval_acc = correct_nodes / total_nodes
    # print(f'Eval Acc: {eval_acc:.4f}')

    return eval_acc


best_eval_acc = 0
best_epoch = 0
flag = 0
for epoch in range(1, epoch+1):
    print('epoch:', epoch)
    train_acc, train_loss = train()
    if epoch%10 == 0:
        torch.save(model.state_dict(), checkpoints_path + 'latest.pt')
        eval_acc = validation(eval_loader)
        writer.add_scalars('Body Segmentation train', {'validation accuracy': eval_acc}, epoch)
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoints_path + 'best.pt')
        print(f'Best Eval Acc: {best_eval_acc:.4f}, Best Epoch: {best_epoch}')

    if train_acc > 0.8:
        flag = 1
        
    if flag == 1:
        scheduler.step()

    writer.add_scalars('Body Segmentation train',
                       {'training loss': train_loss,
                        'training accuracy': train_acc}, epoch)    
