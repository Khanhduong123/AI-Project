import torchvision.transforms as T


def build_transforms(is_train=True):
    list_transforms = []

    if is_train:
        list_transforms.extend([
            T.Resize((32, 32)),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip()
        ])
    else:
        list_transforms.append(
            T.Resize((32, 32))
        )
    
    list_transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform = T.Compose(list_transforms)
    return transform
