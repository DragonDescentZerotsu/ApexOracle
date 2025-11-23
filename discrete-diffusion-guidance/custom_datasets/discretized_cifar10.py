import einops
import torch
import torchvision
from PIL import Image


class DummyVisionTokenizer:
  def __init__(self, vocab_size, image_size,
               add_mask_token=True,
               add_special_tokens=True):
    self.pad_token_id = None
    self.pad_token = None
    if add_mask_token:
      self.mask_token = vocab_size
      self.mask_token_id = vocab_size
      self.vocab_size = vocab_size + 1  # mask token
    else:
      self.vocab_size = vocab_size
    if add_special_tokens:
      self.bos_token_id = vocab_size
      self.bos_token = vocab_size
      self.eos_token_id = vocab_size + 1
      self.eos_token = vocab_size + 1
      self.vocab_size = self.vocab_size + 2  # mask token, bos_token, eos_token
    else:
      self.vocab_size = self.vocab_size
    self.image_size = image_size

  def __call__(self, x):
    return x

  def batch_decode(self, x):
    return einops.rearrange(x, "b (c h w) -> b c h w", c=3,
                     h=self.image_size)

  def decode(self, x):
    return einops.rearrange(x, "(c h w) -> c h w", c=3,
                     h=self.image_size)


class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
  def __init__(self, root, train, **kwargs):
    super().__init__(root=root, train=train,
                     **kwargs)
    self.transform = torchvision.transforms.Compose(
      [
        torchvision.transforms.Resize(32),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
          lambda x: einops.rearrange(x, "c h w -> (c h w)")),
      ]
    )

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)
    img = (img * 255).to(torch.long)

    if self.target_transform is not None:
      target = self.target_transform(target)

    attention_mask = torch.ones_like(img)

    return {'input_ids': img, 'labels': target,
            'attention_mask': attention_mask}
