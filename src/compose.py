"""Side-by-side frame composition used by the Gradio video callback.

Layout:  [ BEV (square, target_h × target_h) ][ camera view (aspect-preserved) ]
with a small header bar across the top for labels.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _default_font(size=16):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def compose_side_by_side(bev_img, cam_img, bev_label, cam_label, target_h=360):
    """Return an RGB uint8 ndarray with BEV and camera view aligned."""
    bev_pil = Image.fromarray(bev_img).resize((target_h, target_h), Image.BILINEAR)

    ch, cw = cam_img.shape[:2]
    new_cw = max(1, int(round(cw * target_h / ch)))
    cam_pil = Image.fromarray(cam_img).resize((new_cw, target_h), Image.BILINEAR)

    header = 32
    total_w = target_h + new_cw
    canvas = Image.new("RGB", (total_w, target_h + header), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    font = _default_font(size=16)
    draw.text((10, 8), bev_label, fill=(230, 230, 230), font=font)
    draw.text((target_h + 10, 8), cam_label, fill=(230, 230, 230), font=font)
    canvas.paste(bev_pil, (0, header))
    canvas.paste(cam_pil, (target_h, header))
    return np.array(canvas)
