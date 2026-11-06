import os
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms


class TransformedSubset(Dataset):
    """
    Wrapper peste torch.utils.data.Subset care permite aplicarea
    unui set distinct de transformari pentru fiecare split in parte
    """
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def get_data_transforms() -> Dict[str, transforms.Compose]:
    """
    Defineste transformarile aplicate imaginilor.
    Imaginile sunt aduse la rezolutia 224x224 si normalizate cu mediile ImageNet.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    }
    return data_transforms


def build_datasets(
    data_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    """
    Incarca imaginile din folder si le imparte in train/val/test conform proportiilor specificate.
    """
    if not (round(train_ratio + val_ratio + test_ratio, 2) == 1.0):
        raise ValueError("Suma proportiilor (train + val + test) trebuie sa fie egala cu 1.0.")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directorul specificat nu exista: {data_dir}")

    # Incarcam dataset-ul brut fara transformari (pentru a aplica transformari distincte pe fiecare subset)
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names = base_dataset.classes

    total_samples = len(base_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    # Split reproductibil folosind seed fix
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    data_transforms = get_data_transforms()

    train_dataset = TransformedSubset(train_subset, transform=data_transforms['train'])
    val_dataset = TransformedSubset(val_subset, transform=data_transforms['val'])
    test_dataset = TransformedSubset(test_subset, transform=data_transforms['test'])

    return train_dataset, val_dataset, test_dataset, class_names


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Returneaza DataLoader-ele pentru train, val si test.
    """
    train_dataset, val_dataset, test_dataset, class_names = build_datasets(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    print("[INFO] Verificare pipeline de date si split-uri...")

    # Calea directa catre folderul 'raw'
    possible_paths = [
        os.path.join("data", "raw"),
        os.path.join("..", "data", "raw")
    ]

    dataset_path = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = p
            break

    if dataset_path is None:
        print("[ATENTIE] Directorul 'data/raw' nu a fost gasit.")
    else:
        print(f"[OK] Setul de date a fost identificat la: {dataset_path}")
        train_loader, val_loader, test_loader, classes = get_data_loaders(
            data_dir=dataset_path,
            batch_size=32,
            num_workers=2
        )

        print(f"[OK] Clase identificate ({len(classes)}): {classes}")
        print(f"[OK] Mostre antrenare (Train): {len(train_loader.dataset)} imagini (~70%)")
        print(f"[OK] Mostre validare (Val): {len(val_loader.dataset)} imagini (~15%)")
        print(f"[OK] Mostre testare (Test): {len(test_loader.dataset)} imagini (~15%)")

        # Extragere a unui batch de test
        images, labels = next(iter(train_loader))
        print(f"[OK] Dimensiune batch imagini: {images.shape} (Batch, Canale, Inaltime, Latime)")
        print(f"[OK] Dimensiune batch etichete: {labels.shape}")
        print("[SUCCES] Dataset-ul si DataLoader-ele sunt complet functionale conform cerintelor S6!")