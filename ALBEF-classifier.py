import torch
import torchvision.transforms as t
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors.blip_processors import BlipCaptionProcessor

from torch.utils.data import DataLoader
from torchvision.datasets.EmoSet2 import EmoSet2

# Pipeline setup
print("imports complete")

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("test 1")

model, vis_processors, _ = load_model_and_preprocess("albef_feature_extractor", model_type="base",
                                                     is_eval=True, device=device)

print("test 2 (model loaded)")

cls_names = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]

text_processor = BlipCaptionProcessor(prompt="This image contains the emotion of ")

cls_prompt = [text_processor(cls_nm) for cls_nm in cls_names]

print(cls_prompt)
print("test 3")


def img_pipeline(img):
    # raw_image = Image.open(img_path).convert("RGB")
    # print(raw_image)

    transform = t.ToPILImage()
    raw_image = transform(img)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # print("test 4")

    sample = {"image": image, "text_input": cls_names}

    image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0]
    text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0]

    # print(image_features)
    # print(text_features)

    sims = (image_features @ text_features.t())[0] / model.temp
    probs = torch.nn.Softmax(dim=0)(sims).tolist()
    # print("test 6")

    max_idx = probs.index(max(probs))
    # print(img_path)
    # for cls_nm, prob in zip(cls_names, probs):
    #     print(f"{cls_nm}: \t {prob:.3%}")
    # print("Max Val: " + cls_prompt[max_idx])
    # print("____________")

    return max_idx


# print("Sadness 1 output: " + img_pipeline(r"C:\Users\manee\Downloads\EmoSet-118K-7\image\sadness\sadness_00000.jpg"))
# print("Sadness 2 output: ")

data_root = r"C:\Users\manee\Downloads\EmoSet-118K-7"
num_emotion_classes = 8
phase = 'val'
# print(data_root)

dataset = EmoSet2(
    data_root=data_root,
    num_emotion_classes=num_emotion_classes,
    phase=phase,
)

info = dataset.get_info(data_root, num_emotion_classes)
# print(dataset.info)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# METRICS
count = 0
hasEmotion = 0
emotions = [0, 0, 0, 0, 0, 0, 0, 0]

log_sep = 50
log_num = 0

scores = {
    0: {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'precision': 0,
        'recall': 0,
        'F1': 0,
    },
    1: {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'precision': 0,
        'recall': 0,
        'F1': 0,
    },
    2: {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'precision': 0,
        'recall': 0,
        'F1': 0,
    },
    3: {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'precision': 0,
        'recall': 0,
        'F1': 0,
    },
    4: {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'precision': 0,
        'recall': 0,
        'F1': 0,
    },
    5: {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'precision': 0,
        'recall': 0,
        'F1': 0,
    },
    6: {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'precision': 0,
        'recall': 0,
        'F1': 0,
    },
    7: {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'precision': 0,
        'recall': 0,
        'F1': 0,
    }
}


def update_scores(actual, pred):
    for i in scores.keys():
        if i not in (actual, pred):
            scores[i]['TN'] += 1

    if actual == pred:
        scores[actual]['TP'] += 1
        scores[pred]['TP'] += 1

    else:
        scores[actual]['FN'] += 1
        scores[pred]['FP'] += 1


def get_final_scores():
    for emo_id in scores.keys():
        try:
            scores[emo_id]['precision'] = scores[emo_id]['TP'] / (scores[emo_id]['TP'] + scores[emo_id]['FP'])
            scores[emo_id]['recall'] = scores[emo_id]['TP'] / (scores[emo_id]['TP'] + scores[emo_id]['FN'])
            scores[emo_id]['F1'] = ((2 * scores[emo_id]['precision'] * scores[emo_id]['recall']) /
                                    (scores[emo_id]['precision'] + scores[emo_id]['recall']))
        except ZeroDivisionError:
            print("ZeroDivisionError for emotion " + str(emo_id))
            continue


def final_scores():
    text = f"""
            # Amusement: {str(emotions[0])}
            Amusement Detected: {scores[0]['TP'] + scores[0]['FP']}
            Amusement Precision: {scores[0]['precision']}
            Amusement Recall: {scores[0]['recall']}
            Amusement F1: {scores[0]['F1']}

            # Awe: {str(emotions[1])}
            Awe Detected: {scores[1]['TP'] + scores[1]['FP']}
            Awe Precision: {scores[1]['precision']}
            Awe Recall: {scores[1]['recall']}
            Awe F1: {scores[1]['F1']}

            # Contentment: {str(emotions[2])}
            Contentment Detected: {scores[2]['TP'] + scores[2]['FP']}
            Contentment Precision: {scores[2]['precision']}
            Contentment Recall: {scores[2]['recall']}
            Contentment F1: {scores[2]['F1']}

            # Excitement: {str(emotions[3])}
            Excitement Detected: {scores[3]['TP'] + scores[3]['FP']}
            Excitement Precision: {scores[3]['precision']}
            Excitement Recall: {scores[3]['recall']}
            Excitement F1: {scores[3]['F1']}

            # Anger: {str(emotions[4])}
            Anger Detected: {scores[4]['TP'] + scores[4]['FP']}
            Anger Precision: {scores[4]['precision']}
            Anger Recall: {scores[4]['recall']}
            Anger F1: {scores[4]['F1']}

            # Disgust: {str(emotions[5])}
            Disgust Detected: {scores[5]['TP'] + scores[5]['FP']}
            Disgust Precision: {scores[5]['precision']}
            Disgust Recall: {scores[5]['recall']}
            Disgust F1: {scores[5]['F1']}

            # Fear: {str(emotions[6])}
            Fear Detected: {scores[6]['TP'] + scores[6]['FP']}
            Fear Precision: {scores[6]['precision']}
            Fear Recall: {scores[6]['recall']}
            Fear F1: {scores[6]['F1']}

            # Sadness: {str(emotions[7])}
            Sadness Detected: {scores[7]['TP'] + scores[7]['FP']}
            Sadness Precision: {scores[7]['precision']}
            Sadness Recall: {scores[7]['recall']}
            Sadness F1: {scores[7]['F1']}

            # Total: {count}
            Overall Precision: {sum([scores[i]['precision'] for i in scores.keys()]) / 8}
            Overall Recall: {sum([scores[i]['recall'] for i in scores.keys()]) / 8}
            Overall F1: {sum([scores[i]['F1'] for i in scores.keys()]) / 8}
            """
    return text


for i, data in enumerate(dataloader):
    # Assuming data['image'] has the shape (1, 3, 224, 224)
    # print(data['image'])
    # if i < 100:
    img = data['image'][0]  # Remove the batch dimension, shape becomes (3, 224, 224)
    emo_idx_actual = data['emotion_label_idx'].item()
    emo_idx_pred = img_pipeline(img)

    update_scores(emo_idx_actual, emo_idx_pred)

    # emotion = info['emotion']['idx2label'][str(data['emotion_label_idx'].item())]
    # print("Emotion: " + emotion)
    emotions[emo_idx_actual] += 1

    # Alt Attr processing
    """if str(data['scene_label_idx'].item()) != "-1":
        scene = info['scene']['idx2label'][str(data['scene_label_idx'].item())]
        # print("Scene: " + scene)

    if str(data['facial_expression_label_idx'].item()) != "-1":
        facial_expression = info['facial_expression']['idx2label'][str(data['facial_expression_label_idx'].item())]
        # print("Facial Expression: " + facial_expression)

    if str(data['human_action_label_idx'].item()) != "-1":
        human_action = info['human_action']['idx2label'][str(data['human_action_label_idx'].item())]
        # print("Human Action: " + human_action)

    if str(data['brightness_label_idx'].item()) != "-1":
        brightness = str(info['brightness']['idx2label'][str(data['brightness_label_idx'].item())])
        # print("Brightness: " + brightness)

    if str(data['colorfulness_label_idx'].item()) != "-1":
        colorfulness = str(info['colorfulness']['idx2label'][str(data['colorfulness_label_idx'].item())])
        # print("Colorfulness: " + colorfulness)

    objs_present = []
    for idx, val in enumerate(data['object_label_idx'][0].numpy()):
        if val == 1:
            object = info['object']['idx2label'][str(idx)]
            objs_present.append(object)
    if len(objs_present) > 0:
        pass
        # print("Objects Present:")
        # print(objs_present)
    """

    count += 1

    if i % log_sep == 0:
        log_num += 1
        print("Log Num: " + str(log_num))
        print("Count: " + str(count))
    # else:
    #     break

print("getting final scores")
get_final_scores()
print(final_scores())
print("Scores: ")
print(scores)
# print("Count: " + str(count))
# print("# Amusement: " + str(emotions[0]))
# print("# Awe: " + str(emotions[1]))
# print("# Contentment: " + str(emotions[2]))
# print("# Excitement: " + str(emotions[3]))
# print("# Anger: " + str(emotions[4]))
# print("# Disgust: " + str(emotions[5]))
# print("# Fear: " + str(emotions[6]))
# print("# Sadness: " + str(emotions[7]))
