import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def predict(sample, model, scaler_obj=StandardScaler(),device="cpu"):
    """
    Make a prediction on a given sample with a model.

    Parameters
    ----------
    sample : array-like
        Input sample to be predicted.
    model : nn.Module
        Model to make prediction with.
    scaler_obj : object, optional
        Scaler object to scale the input data, by default StandardScaler()

    Returns
    -------
    array-like
        Prediction probabilities.
    """
    model.eval()
    with torch.no_grad():
        sample_tensor = torch.FloatTensor(scaler_obj.transform([sample])).to(device)
        logits = model(sample_tensor)
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()[0]