import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image

# --------------------------
# 1. Custom Dataset
# --------------------------
class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name + ".json")

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load JSON annotation
        with open(ann_path) as f:
            data = json.load(f)
        boxes = []
        for obj in data["objects"]:
            if obj["classTitle"] != "license plate":
                continue
            x1, y1 = obj["points"]["exterior"][0]
            x2, y2 = obj["points"]["exterior"][1]
            boxes.append([x1, y1, x2, y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # single class = 1

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            img = self.transforms(img)

        return img, target

# --------------------------
# 2. Transformations
# --------------------------
import torchvision.transforms as T
def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

# --------------------------
# 3. Model
# --------------------------
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --------------------------
# 4. Dataset paths
# --------------------------
train_img_dir = r"C:\Users\Malek\Desktop\AutoPlateTN\tunisian-licensed-plates-DatasetNinja\train\img"
train_ann_dir = r"C:\Users\Malek\Desktop\AutoPlateTN\tunisian-licensed-plates-DatasetNinja\train\ann"

val_img_dir = r"C:\Users\Malek\Desktop\AutoPlateTN\tunisian-licensed-plates-DatasetNinja\test\img"
val_ann_dir = r"C:\Users\Malek\Desktop\AutoPlateTN\tunisian-licensed-plates-DatasetNinja\test\ann"

train_dataset = LicensePlateDataset(train_img_dir, train_ann_dir, transforms=get_transform())
val_dataset = LicensePlateDataset(val_img_dir, val_ann_dir, transforms=get_transform())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# --------------------------
# 5. Device
# --------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --------------------------
# 6. Initialize model
# --------------------------
num_classes = 2  # 1 class (plate) + background
model = get_model(num_classes)
model.to(device)

# --------------------------
# 7. Optimizer & LR scheduler
# --------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

# --------------------------
# 8. Training loop
# --------------------------
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Step LR scheduler
    lr_scheduler.step()

    # Optional: evaluation on validation set can be added here

# --------------------------
# 9. Save model
# --------------------------
torch.save(model.state_dict(), "fasterrcnn_tunisia_plates.pth")
print("Training complete, model saved!")
