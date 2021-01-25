import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

import utils
from model import SSD300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = ['background',
          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
          'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
color_map = dict(zip(labels, [None, '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                              '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                              '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']))


# Load model checkpoint
checkpoint_path = 'ssd300.pt'
checkpoint = torch.load(checkpoint_path)
print('Loaded checkpoint from epoch %d.' % checkpoint.get('epoch', 0))
model = SSD300(21)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def detect(image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    original_image = image

    # Transform
    image = normalize(to_tensor(resize(image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [labels[lb] for lb in det_labels[0].to('cpu').tolist() if lb != 0]

    # If no objects found, Just return original image
    if len(det_labels) == 0:
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        color = color_map[det_labels[i]]

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=color)
        draw.rectangle(xy=[lb + 1. for lb in box_location], outline=color)

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=color)
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)

    return annotated_image


if __name__ == '__main__':
    img_path = './data/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
    image = Image.open(img_path, mode='r').convert('RGB')
    detect(image, min_score=0.3, max_overlap=0.5, top_k=200).show()
