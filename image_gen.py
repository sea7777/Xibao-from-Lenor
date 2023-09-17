import cv2
import os
import jieba
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def mlp(suffix):
    return os.path.join(os.path.dirname(__file__), suffix)


def auto_warp(text_list, font, image, force=False):
    def is_too_long(_text):
        bbox = draw.textbbox((0, 0), _text, font=font, align="center")
        width = bbox[2] - bbox[0]
        if width > 3 * image.size[0] // 4:
            return True
        return False

    def is_too_high(_text):
        bbox = draw.textbbox((0, 0), _text, font=font, align="center")
        height = bbox[3] - bbox[1]
        if height > 550:
            return True
        return False

    text_list = text_list.copy()
    draw = ImageDraw.Draw(image)
    result = [""]
    while len(text_list) > 0:
        current = result[-1] + text_list[0]
        if is_too_long(current):
            result.append(text_list.pop(0))
            if is_too_high("\n".join(result)):
                if force:
                    return "\n".join(result[:-1])
                else:
                    return None
        else:
            result[-1] += text_list.pop(0)
    return "\n".join(result)


def multiply_image(img1, img2):
    img1 = img1 / 255
    img2 = img2 / 255
    img1 = img1 * img2
    img1 = img1 * 255
    img1 = img1.astype(np.uint8)
    return img1


def overlay_image(upper, lower):
    upper_gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(upper_gray, 254, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    upper_fg = cv2.bitwise_and(upper, upper, mask=mask)
    lower_bg = cv2.bitwise_and(lower, lower, mask=mask_inv)
    return cv2.add(lower_bg, upper_fg)


def regularize_cut(text_cut):
    ret = []
    for phrase in text_cut:
        while len(phrase) > 20:
            ret.append(phrase[:20])
            phrase = phrase[20:]
        ret.append(phrase)
    return ret


def get_text_mask(text, img):
    shape = img.shape
    img = np.ones(shape, np.uint8) * 255
    center_pos = (shape[1] // 2, shape[0] * 9 // 20)
    font_path = mlp("./assets/font.ttf")
    scales = (96, 64, 48, 32, 24, 16)
    auto_scale = 0
    jieba.setLogLevel(jieba.logging.INFO)
    text_cut = jieba.lcut(text)
    text_cut = regularize_cut(text_cut)
    while True:
        font = ImageFont.truetype(font_path, scales[auto_scale])
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        text_wrap = auto_warp(text_cut, font, img_pil)
        if text_wrap is None:
            if auto_scale < len(scales) - 1:
                auto_scale += 1
            else:
                text_wrap = auto_warp(text_cut, font, img_pil, force=True)
                break
        else:
            break
    bbox = draw.textbbox(center_pos, text_wrap, font=font, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    center_pos = (center_pos[0] - width // 2, center_pos[1] - height // 3)
    draw.text(center_pos, text_wrap, fill=(35, 48, 220), font=font, align="center")
    img = np.asarray(img_pil)
    img_cny = cv2.Canny(img, 100, 200)
    contour = cv2.findContours(img_cny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_ret = np.ones(shape, np.uint8) * 255
    cv2.drawContours(img_ret, contour[0], -1, (0, 215, 255), 5)
    img_ret = multiply_image(img, img_ret)
    return img_ret


def put_text(img, text, gray):
    text = text[:10000]
    text_mask = get_text_mask(text, img)
    if gray:
        text_mask = cv2.cvtColor(text_mask, cv2.COLOR_BGR2GRAY)
        text_mask = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
    img = overlay_image(text_mask, img)
    return img


def xi_bao(text, path=None):
    """
    :param text: text for xi bao
    :param path: path to save image. Won't be ensured,
    :return:
    """
    xi_bao_image = cv2.imread(mlp("assets/xi_bao.webp"))
    xi_bao_image = put_text(xi_bao_image, text, False)
    #cv2.imwrite(path, xi_bao_image)
    if not path:
        cv2.imencode('.webp', xi_bao_image)[1].tofile('output/'+'喜报：'+text+'.webp')
    else:
        cv2.imencode('.webp', xi_bao_image)[1].tofile(path)


def bei_bao(text, path=None):
    """
    :param text: text for xi bao
    :param path: path to save image. Won't be ensured,
    :return:
    """
    bei_bao_image = cv2.imread(mlp("assets/bei_bao.webp"))
    bei_bao_image = put_text(bei_bao_image, text, True)
    #cv2.imwrite(path, bei_bao_image)
    if not path:
        cv2.imencode('.webp', bei_bao_image)[1].tofile('output/'+'悲报：'+text+'.webp')
    else:
        cv2.imencode('.webp', bei_bao_image)[1].tofile(path)


if __name__ == '__main__':
    xi_bao("本群升级为原神群")
