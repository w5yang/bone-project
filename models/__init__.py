from torchvision import transforms

# Choose the transform for the corresponding model. Tiny for 32x32 input, full for 224x224 input.

sized_transforms = {
    32: transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    ),
    224: transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
}

imagenet_transform = {
    32: transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}