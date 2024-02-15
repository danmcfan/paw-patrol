import torch
import torchvision
from torchvision.transforms import Resize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)

from src.data import create_images, Image


class ImageDataset(Dataset):
    def __init__(self, images: list[Image], size=(300, 300)):
        self.images = images
        self.size = size

    def __getitem__(self, idx):
        image: Image = self.images[idx]

        pil_image = image.get_pil_image()
        original_size = pil_image.size
        resize_transform = Resize(self.size)
        resized_pil_image = resize_transform(pil_image)
        tensor_image = torchvision.transforms.ToTensor()(resized_pil_image)

        scale_width = self.size[0] / original_size[0]
        scale_height = self.size[1] / original_size[1]

        boxes = torch.tensor(
            [
                [
                    obj.bounding_box.xmin * scale_width,
                    obj.bounding_box.ymin * scale_height,
                    obj.bounding_box.xmax * scale_width,
                    obj.bounding_box.ymax * scale_height,
                ]
                for obj in image.objects
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([1] * len(image.objects), dtype=torch.int64)

        return {
            "image": tensor_image,
            "boxes": boxes,
            "labels": labels,
        }

    def __len__(self):
        return len(self.images)


def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train(
    filename: str,
    batch_size: int = 16,
    num_epochs: int = 10,
    count: int | None = None,
):
    num_classes = 2
    model = get_model(num_classes)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    images = create_images()

    if count is None or count > len(images):
        count = len(images)

    images = images[:count]

    train_dataset = ImageDataset(images)

    def collate_fn(batch):
        images = [item["image"] for item in batch]
        targets = [
            {"boxes": item["boxes"], "labels": item["labels"]}
            for item in batch
        ]

        images = default_collate(images)
        return images, targets

    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch, (images, targets) in enumerate(data_loader):
            print(f"    Batch {batch + 1}/{len(data_loader)}")

            images = images.to(device)
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"        Loss: {losses.item()}")

    torch.save(model.state_dict(), f"data/models/{filename}.pth")


def predict(filename: str, image: Image, threshold: float = 0.5):
    model = get_model(2)
    model.load_state_dict(torch.load(f"data/models/{filename}.pth"))
    model.eval()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    pil_image = image.get_pil_image()
    original_size = pil_image.size
    resize_transform = Resize((300, 300))
    resized_pil_image = resize_transform(pil_image)
    tensor_image = torchvision.transforms.ToTensor()(resized_pil_image)

    scale_width = 300 / original_size[0]
    scale_height = 300 / original_size[1]

    with torch.no_grad():
        prediction = model([tensor_image.to(device)])
        prediction = prediction[0]

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        box = [
            (
                int(coord / scale_width)
                if i % 2 == 0
                else int(coord / scale_height)
            )
            for i, coord in enumerate(box)
        ]
        if score > threshold:
            yield box, label, score
