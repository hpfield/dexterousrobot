import numpy as np
import torch

from sklearn.metrics import confusion_matrix


class LabelEncoder:

    def __init__(self, label_names, device='cuda'):
        self.device = device
        self.target_label_names = label_names
        self.tolerences = np.ones(len(label_names))

    @property
    def out_dim(self):
        return len(self.target_label_names)

    def encode_label(self, labels_dict):
        """
        Process label data to NN friendly label for prediction.

        Returns: torch tensor that will be predicted by the NN
        """
        return torch.nn.functional.one_hot(labels_dict['id'], num_classes=self.out_dim).float().to(self.device)

    def decode_label(self, outputs):
        """
        Process NN predictions to label data, always decodes to cpu.

        Returns: Dict of np arrays in suitable format for downstream task.
        """

        ids = outputs.argmax(dim=1).detach().cpu().numpy()
        return {
            'id': ids,
            'label': np.array(self.target_label_names)[ids]
        }

    def print_metrics(self, metrics):
        """
        Formatted print of metrics given by calc_metrics.
        """
        conf_mat = metrics['conf_mat']
        print('Accuracy: ')
        print({key: val for key, val in zip(self.target_label_names, np.diag(conf_mat))})

    def write_metrics(self, writer, metrics, epoch, mode='val'):
        """
        Write metrics given by calc_metrics to tensorboard.
        """
        pass

    def calc_metrics(self, labels, predictions):
        """
        Calculate metrics useful for measuring progress throughout training.

        Returns: dict of metrics
            {
                'metric': np.array()
            }
        """
        conf_mat = self.acc_metric(labels, predictions)
        metrics = {
            'conf_mat': conf_mat
        }
        return metrics

    def acc_metric(self, labels, predictions):
        """
        Accuracy metric for classification problem.
        """
        conf_mat = confusion_matrix(predictions['id'], labels['id'])
        return conf_mat.astype('float') / (conf_mat.sum(axis=1)[:, np.newaxis] + 1e-8)
