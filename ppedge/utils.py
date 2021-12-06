#!/usr/bin/env python
# coding: utf-8


def resize_image(img, reference_img, n_channel=3):
    """

    Parameters
    ----------
    img: array.
        3dim (N, N, channel)
    reference_img: array with target image
    n_channel: int


    Returns
    -------
    img: np.ndarray. 3dims

    """
    if len(img.shape) != 3 or len(reference_img.shape) != 3:
        raise ValueError(
            "image shape must be 3 dim, however, input: {}".format(img.shape)
        )

    # Resizing
    if img.shape[0:2] != reference_img.shape[0:2]:
        img = cv2.resize(img, dsize=(reference_img.shape[0:2]))

    # channel wise padding
    if n_channel >= 2:
        img = np.stack([img for _ in range(n_channel)])
        img = np.transpose(img, axes=[1, 2, 0])
    return img
