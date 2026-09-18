from torchvision import transforms


def get_train_transforms(image_size: tuple[int, int] = (64, 384)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomAffine(
            degrees=2,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transforms(image_size: tuple[int, int] = (64, 384)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
