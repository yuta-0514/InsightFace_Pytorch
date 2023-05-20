import torchvision.transforms as T


transform = T.Compose(
    [
     T.RandomHorizontalFlip(p=0.3),
     T.RandomAffine(degrees=[-30, 30], translate=(0.2, 0.1), scale=(1.0, 1.5)),
     T.ColorJitter(brightness=0.5),
     T.GaussianBlur(kernel_size=3),
     ]
)
