import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from torchvision import transforms


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    processor: Optional[ProcessorMixin] = None

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor is not None:
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.tokenizer
        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        input_ids = inputs.pop("input_ids")
        input_ids = [input_id.squeeze(0) for input_id in input_ids]
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        inputs.pop("attention_mask")
        batched_inputs = {}
        for key, values in inputs.items():
            # 对非张量类型（如 raw_images 的 PIL 列表）直接保留列表，不做 concat
            if not isinstance(values[0], torch.Tensor):
                batched_inputs[key] = values
            else:
                batched_inputs[key] = torch.concatenate(values, dim=0)
        batched_inputs["input_ids"] = input_ids
        batched_inputs["attention_mask"] = attention_mask

        return batched_inputs


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Union[HFDataset, str],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        text_key: str,
        image_key: Optional[str] = None,
        video_key: Optional[str] = None,
        audio_key: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(dataset, str):
            dataset = HFDataset.from_parquet(dataset)

        self.tokenizer = tokenizer
        self.processor = processor
        self.image_key = image_key
        self.video_key = video_key
        self.audio_key = audio_key
        self.text_key = text_key
        self.dataframe = dataset

    def __getitem__(self, index):
        row = self.dataframe[index]
        # print(row)
        """
{'images': [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x333 at 0x7F5904214FC0>], 'text': [{'content': [{'text': None, 'type': 'image'}, {'text': '\nWhat do you think is the person to the right of the worker wearing?\nAnswer the question using a single w
ord or phrase.', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Coat', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': 'Is the fence to the left of the train running or standing?', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Runn
ing', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': 'Is the weather overcast or sunny today?', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Overcast', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': 'Is it an outdoors scene?'
, 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Yes', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': 'Are there any boats?', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'No', 'type': 'text'}], 'role': 'assistant'}, {'content': [
{'text': 'Is the gray sky cloudless or overcast?', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Overcast', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': 'Is there a fence to the right of the vehicle in the middle of the image?', 'type': 't
ext'}], 'role': 'user'}, {'content': [{'text': 'Yes', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': 'Is the train to the right or to the left of the person that is wearing a coat?', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Left', 'type
': 'text'}], 'role': 'assistant'}, {'content': [{'text': 'What is the vehicle that is to the right of the fence on the left?', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Train', 'type': 'text'}], 'role': 'assistant'}]}                                   
{'images': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=640x480 at 0x7F5904214FC0>], 'text': [{'content': [{'text': None, 'type': 'image'}, {'text': '\nProvide a short description for the given region.\n[0.408,0.652,0.869,0.856]', 'type': 'text'}], 'role': 'u
ser'}, {'content': [{'text': 'Papers sitting on side of desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.586,0.673,0.736,0.750]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Blue mousepad sitting on desk.', 'type': 'text'}], 'rol
e': 'assistant'}, {'content': [{'text': '[0.583,0.673,0.736,0.752]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Mouse sitting on top of mousepad.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.214,0.690,0.316,0.996]', 'type': 'text'
}], 'role': 'user'}, {'content': [{'text': 'Wire hanging off of desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.002,0.500,0.073,0.681]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Open tissue box sitting on desk.', 'type': 'tex
t'}], 'role': 'assistant'}, {'content': [{'text': '[0.603,0.481,0.662,0.627]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Speaker with power light on.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.330,0.610,0.580,0.688]', 'type': '
text'}], 'role': 'user'}, {'content': [{'text': 'Black keyboard sitting on desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.458,0.548,0.480,0.577]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Power button for computer monitor.',
 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.398,0.560,0.445,0.579]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Group of buttons on front of monitor.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.766,0.750,0.
867,0.848]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Paper on the desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.011,0.671,0.108,0.775]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Paper on the desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.645,0.669,0.691,0.733]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Mouse on the mousepad.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.595,0.481,0.672,0.629]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Speaker on the desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.250,0.465,0.323,0.598]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Speaker on the desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.319,0.606,0.598,0.694]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Keyboard on te desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.339,0.237,0.608,0.615]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'Monitor on the desk.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.328,0.594,0.594,0.708]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'A black keyboard.', 'type': 'text'}], 'role': 'assistant'}, {'conte
nt': [{'text': '[0.588,0.652,0.734,0.752]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'A black mouse on a blue pad.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.584,0.652,0.734,0.758]', 'type': 'text'}], 'role': 'user'}, {'content
': [{'text': 'A mouse on a blue mouse pad.', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '[0.336,0.273,0.675,0.627]', 'type': 'text'}], 'role': 'user'}, {'content': [{'text': 'A speaker next to a monitor.', 'type': 'text'}], 'role': 'assistant'}]}
        """

        if self.processor is not None:
            # By default we assume
            text = self.processor.apply_chat_template(
                row[self.text_key], tokenize=False, add_generation_prompt=False
            )
            
            multi_modal_inputs = {}
            images = None
            if self.image_key in row:
                images = [process_image(image) for image in row[self.image_key]]
                multi_modal_inputs["images"] = images
            # print(images)
            # TODO
            # Implement the load logic for video and audios later
            # if self.video_key in row:
            #     videos = [video for video in row[self.video_key]]
            #     multi_modal_inputs["videos"] = videos

            # if self.audio_key in row:
            #     audios = [audio for audio in row[self.audio_key]]
            #     multi_modal_inputs["audios"] = audios 

            model_inputs = self.processor(
                text=[text], return_tensors="pt", 
                truncation=True,
                max_length=2048, 
                **multi_modal_inputs
            )
            # print(text)
            # print(model_inputs)
            model_inputs['raw_images'] = images
        else:
            text = self.tokenizer.apply_chat_template(
                row[self.text_key], tokenize=False, add_generation_prompt=False
            )
            model_inputs = self.tokenizer([text], return_tensors="pt", 
                                          truncation=True, max_length=2048)
        
        return model_inputs

    def get_collator(self):
        return DataCollator(self.tokenizer, self.processor)

    def __len__(self):
        return len(self.dataframe)


def process_image(img):
    # img: PIL.Image or torch.Tensor (C, H, W)

    w, h = img.size
    if w > 504 or h > 504:          # resize img if too large.  504, 308
        img = transforms.Resize([504, 504])(img)
        # img =  transforms.CenterCrop(896)(img)
    return img
