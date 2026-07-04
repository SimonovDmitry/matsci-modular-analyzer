import cv2
import numpy as np
import torch


class TileProcessor:
    def __init__(self, tile_size=512, overlap=64):
        if overlap >= tile_size:
            raise ValueError(f"Overlap ({overlap}) must be less than tile_size ({tile_size})")

        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

        self.tiles = None
        self.coords = None
        self.h = None
        self.w = None

    def split_into_tiles(self, image):
        self.h, self.w = image.shape[:2]
        tiles = []
        coords = []
        y_starts = []
        y = 0

        while y + self.tile_size <= self.h:
            y_starts.append(y)
            y += self.stride
        if y_starts and y_starts[-1] + self.tile_size < self.h:
            y_starts[-1] = self.h - self.tile_size

        x_starts = []
        x = 0
        while x + self.tile_size <= self.w:
            x_starts.append(x)
            x += self.stride
        if x_starts and x_starts[-1] + self.tile_size < self.w:
            x_starts[-1] = self.w - self.tile_size

        for y in y_starts:
            for x in x_starts:
                tile = image[y:y + self.tile_size, x:x + self.tile_size]
                tiles.append(tile)
                coords.append((x, y))
        self.tiles = np.array(tiles)
        self.coords = coords
        return self.tiles, self.coords

    def tiles_to_tensor(self, data, device='cpu'):
        if data.ndim == 4 and data.shape[3] == 3:
            img_rgb = data[..., ::-1].copy()
        else:
            img_rgb = data.copy()

        tensor = torch.from_numpy(img_rgb).permute(0, 3, 1, 2).float() / 255.0
        return tensor.to(device)

    def stitch_tiles(self, predictions):
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()

        n_channels = predictions.shape[1] if predictions.ndim == 4 else 1

        if n_channels == 1:
            result = np.zeros((self.h, self.w), dtype=np.float32)
        else:
            result = np.zeros((self.h, self.w, n_channels), dtype=np.float32)

        weight_map = np.zeros((self.h, self.w), dtype=np.float32)

        for idx, (x, y) in enumerate(self.coords):
            pred = predictions[idx]

            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred[0]
            elif pred.ndim == 3 and n_channels > 1:
                pred = np.transpose(pred, (1, 2, 0))

            weight = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
            if self.overlap > 0:
                ramp = np.linspace(0, 1, self.overlap)
                weight[:self.overlap, :] *= ramp[:, None]
                weight[-self.overlap:, :] *= ramp[::-1, None]
                weight[:, :self.overlap] *= ramp[None, :]
                weight[:, -self.overlap:] *= ramp[None, ::-1]

            result[y:y + self.tile_size, x:x + self.tile_size] += pred * weight
            weight_map[y:y + self.tile_size, x:x + self.tile_size] += weight

        weight_map = np.maximum(weight_map, 1e-8)
        result = result / weight_map

        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result