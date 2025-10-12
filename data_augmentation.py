from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import random
from PIL import Image
import numpy as np
import albumentations as A

class HFlip:
    def __call__(self, img, mask):
        return F.hflip(img), F.hflip(mask)


class VFlip:
    def __call__(self, img, mask):
        return F.vflip(img), F.vflip(mask)


class Rotate:
    def __init__(self, degrees=10):
        self.degrees = degrees
    def __call__(self, img, mask):
        a = random.uniform(-self.degrees, self.degrees)
        return (
            F.rotate(img,  a, interpolation=InterpolationMode.BILINEAR),
            F.rotate(mask, a, interpolation=InterpolationMode.NEAREST),
        )


class RandomAffine:
    def __init__(self, degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05),
                 shear=0.0, fill_img=0, fill_mask=0, center=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill_img = fill_img
        self.fill_mask = fill_mask
        self.center = center  # (x, y) or None

    def _sample_params(self, w, h):
        angle = random.uniform(-self.degrees, self.degrees)

        max_dx = self.translate[0] * w
        max_dy = self.translate[1] * h
        trans = (int(round(random.uniform(-max_dx, max_dx))),
                 int(round(random.uniform(-max_dy, max_dy))))

        sc = random.uniform(self.scale[0], self.scale[1])

        if isinstance(self.shear, (tuple, list)):
            if len(self.shear) == 2 and not isinstance(self.shear[0], (tuple, list)):
                shear = (random.uniform(-self.shear[0], self.shear[0]),
                         random.uniform(-self.shear[1], self.shear[1]))
            elif len(self.shear) == 2 and isinstance(self.shear[0], (tuple, list)):
                sx = random.uniform(self.shear[0][0], self.shear[0][1])
                sy = random.uniform(self.shear[1][0], self.shear[1][1])
                shear = (sx, sy)
            else:
                raise ValueError("Invalid shear specification")
        else:
            shear = (random.uniform(-self.shear, self.shear), 0.0)

        return angle, trans, sc, shear

    def __call__(self, img, mask):
        w, h = img.size
        angle, translate, scale, shear = self._sample_params(w, h)

        img_a = F.affine(
            img, angle=angle, translate=translate, scale=scale, shear=shear,
            interpolation=InterpolationMode.BILINEAR, fill=self.fill_img, center=self.center
        )
        mask_a = F.affine(
            mask, angle=angle, translate=translate, scale=scale, shear=shear,
            interpolation=InterpolationMode.NEAREST, fill=self.fill_mask, center=self.center
        )
        return img_a, mask_a


class Elastic2D:
    def __init__(self, alpha=35, sigma=6, alpha_affine=6, p=1.0):
        self.transform = A.Compose([A.ElasticTransform(alpha=alpha, sigma=sigma,
                                                alpha_affine=alpha_affine,
                                                interpolation=1,   
                                                border_mode=0, p=p)])

    def __call__(self, img, mask):
        img  = np.array(img)
        mask = np.array(mask)
        out = self.transform(image=img, mask=mask)
        return Image.fromarray(out["image"]), Image.fromarray(out["mask"])