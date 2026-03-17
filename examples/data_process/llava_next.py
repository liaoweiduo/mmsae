import os
import argparse
import math
from datasets import load_dataset
from PIL import Image, ImageFile


def convert_llava_next_dataset(example):
    """Convert LLaVA dataset format to the expected format."""
    try:
        img = example.get("image")
        images = [img]
        conversations = example.get("conversations", [])
        if not isinstance(img, (Image.Image, ImageFile.ImageFile)):
            # if images:
            #     print(f"Sample type is {type(img)}")
            # else:
            #     print(f"Sample images is empty")     
            img = Image.new("RGB", (224, 224), color=(255, 255, 255))  # new a empty img
            text = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': ""}]}]
            return {"images": [img], "text": text, "is_valid": False}
        if not conversations:
            print(f"conversations is empty, type: {type(conversations)}, content: {conversations}")
            return {"images": images, "text": ["null"], "is_valid": False}
        messages = []
        for conversation in conversations:
            role = conversation["from"]
            content = conversation["value"]
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            if "<image>" in content:
                content = content.replace("<image>", "")
                messages.append({
                    "role": role,
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": content},
                    ],
                })
            else:
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
        return {"images": images, "text": messages, "is_valid": True}
    except Exception as e:
        print(f"Error: {e} in example {example}")
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))  # new a empty img
        text = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': ""}]}]
        return {"images": [img], "text": text, "is_valid": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_path", type=str, default="/data3/datasets/llavanext"
    )
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset to process")
    parser.add_argument("--batch_size", type=int, default=100_000, help="Batch size for processing")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes for map")
    args = parser.parse_args()

    dataset = load_dataset(
        "lmms-lab/LLaVA-NeXT-Data", split=args.split, cache_dir=args.local_path
    )

    total = len(dataset)
    batch_size = args.batch_size
    num_batches = math.ceil(total / batch_size)
    output_files = []
    # import pdb; pdb.set_trace()
    for i in range(num_batches):        # (num_batches,num_batches):
        out_file = f"{args.local_path}/next_part_{i}.parquet"
        start = i * batch_size
        end = min((i + 1) * batch_size, total-1)
        print(f"Processing batch {i+1}/{num_batches}: {start} ~ {end}")
        if os.path.exists(out_file):
            print(f"Output file {out_file} already exists. Skipping and checking...")
            # _dataset = load_dataset(
            #     "parquet",
            #     data_files=f"{args.local_path}/next_part_{i}.parquet",
            #     cache_dir=args.local_path
            # )["train"]
            # sample = _dataset[0]
            # print(sample)
            # # import pdb;pdb.set_trace()
            # for k, v in sample.items():
            #     print(f"{k}: {type(v)}")

            # print(f"Dataset size: {len(_dataset)} samples.")
            continue
        
        sub_dataset = dataset.select(range(start, end))
        original_columns = sub_dataset.column_names
        sub_dataset = sub_dataset.map(
            convert_llava_next_dataset,
            remove_columns=original_columns,
            num_proc=args.num_proc
        )
        # filter out invalid samples
        sub_dataset = sub_dataset.filter(
            lambda x: x.get("is_valid", True),
            num_proc=args.num_proc
        )
        # remove_columns: `is_valid`
        sub_dataset = sub_dataset.remove_columns(["is_valid"])
        sub_dataset.to_parquet(out_file)
        output_files.append(out_file)
        print(f"Batch {i+1} saved to {out_file}")

    print("All batches processed and saved.")
    print("You can merge them with:")
    print(f"load_dataset('parquet', data_files={output_files})")
    print("And then filter out invalid samples:")
    # print("dataset = dataset.filter(lambda x: x['text'] is not None and x['images'] is not None)")

    dataset = load_dataset(
        "parquet",
        data_files=f"{args.local_path}/next_part_*.parquet",
        cache_dir=args.local_path   
    )["train"]



    sample = dataset[0]
    print(sample)
    # import pdb;pdb.set_trace()
    for k, v in sample.items():
        print(f"{k}: {type(v)}")

    print(f"Dataset size: {len(dataset)} samples")
    for idx, sample in enumerate(dataset):
        images = sample.get("images", [])
        if images and isinstance(images[0], (Image.Image, ImageFile.ImageFile)):
            # print(f"Sample {idx}: images[0] is a valid ImageFile or PIL.Image.Image")
            pass
        else:
            if images:
                print(f"Sample {idx}: images[0] type is {type(images[0])}")
            else:
                print(f"Sample {idx}: images is empty")

    print("Finish!")
    