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
            img_rgb = data[..., ::-1]
        else:
            img_rgb = data
        tensor = torch.from_numpy(img_rgb).permute(0, 3, 1, 2).float() / 255.0
        return tensor.to(device)

    def stitch_tiles(self, predictions):
        predictions = predictions.detach().cpu.numpy()

        if predictions.ndim == 4:
            n_channels = predictions.shape[1]
        else:
            n_channels = 1

        if n_channels == 1:
            result = np.zeros((self.h, self.w), dtype=np.float32)
        else:
            result = np.zeros((self.h, self.w, n_channels), dtype=np.float32)

        weight_map = np.zeros((self.h, self.w), dtype=np.float32)

        for idx, (x, y) in enumerate(self.coords):
            pred = predictions[idx]

            if pred.ndim == 4:
                pred = pred.squeeze(0)

            weight = np.ones((self.tile_size, self.tile_size), dtype=np.float32)

            if x > 0:
                left_fade = np.linspace(0, 1, self.overlap)
                weight[:, :self.overlap] *= left_fade

            if x + self.tile_size < self.w:
                right_fade = np.linspace(1, 0, self.overlap)
                weight[:, -self.overlap:] *= right_fade

            if y > 0:
                top_fade = np.linspace(0, 1, self.overlap).reshape(-1, 1)
                weight[:self.overlap, :] *= top_fade

            if y + self.tile_size < self.h:
                bottom_fade = np.linspace(1, 0, self.overlap).reshape(-1, 1)
                weight[-self.overlap:, :] *= bottom_fade

            if n_channels == 1:
                result[y:y + self.tile_size, x:x + self.tile_size] += pred * weight
            else:
                for c in range(n_channels):
                    result[y:y + self.tile_size, x:x + self.tile_size, c] += pred[:, :, c] * weight

            weight_map[y:y + self.tile_size, x:x + self.tile_size] += weight

        weight_map = np.maximum(weight_map, 1e-8)
        if n_channels == 1:
            result = result / weight_map
        else:
            for c in range(n_channels):
                result[:, :, c] /= weight_map

        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return result

