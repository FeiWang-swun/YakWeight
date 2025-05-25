import torch
from torch.utils.data import DataLoader
from engine import train_one_epoch
from utils import Yakdata
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn
from utils import collate_fn
device = torch.device('cuda')
device = torch.device('cuda')
dataset_train = Yakdata(mode="train")
data_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn)
dataset_test = Yakdata(mode="val")
data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=True, collate_fn=collate_fn)
def get_model(num_keypoints):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                       aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = keypointrcnn_resnet50_fpn(
        pretrained=False,
        pretrained_backbone=True,
        num_keypoints=num_keypoints,
        num_classes=2,  # 背景 + 目标
        rpn_anchor_generator=anchor_generator
    )
    return model
model = get_model(num_keypoints=4)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
    lr_scheduler.step()
    torch.save(model.state_dict(), f'keypointsrcnn_weights_{epoch}.pth')
