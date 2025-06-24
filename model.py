from transformers import ViTForImageClassification

def get_model(model_name_or_path, label_names):
    return ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label_names),
        id2label={str(i): c for i, c in enumerate(label_names)},
        label2id={c: str(i) for i, c in enumerate(label_names)}
    )