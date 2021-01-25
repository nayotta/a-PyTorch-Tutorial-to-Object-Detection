import torch
from model import SSD300
from datasets import PascalVOCDataset
import utils

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
checkpoint_path = 'ssd300.pt'

# Load test data
test_dataset = PascalVOCDataset('./data', split='TEST', keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
n_classes = len(test_dataset.label_map())

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint_path)
model = SSD300(n_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Switch to eval mode
model.eval()


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)

        # Calculate mAP
        APs, mAP = utils.calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, n_classes)

    labels = dict((v, k) for k, v in test_dataset.label_map().items())
    # Print AP for each class
    for i, ap in enumerate(APs):
        print('%s: \t%.4f' % (labels[i + 1], ap))

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)
