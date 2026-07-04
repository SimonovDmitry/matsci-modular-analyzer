import cv2


def normalize_illumination(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return result


def denoise(image, denoise_method='median', kernel_size=3, d=9, sigma_color=75,
            sigma_space=75, h=10):
    if denoise_method == 'median':
        if kernel_size % 2 == 0:
            kernel_size += 1
        res_image = cv2.medianBlur(image, kernel_size)
    elif denoise_method == 'bilateral':
        res_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    elif denoise_method == 'nlm':
        res_image =  cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    else:
        res_image = image
    return res_image
