import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict, Value, ClassLabel


def normalize_columns(ds):
    if "sentence" in ds.column_names:
        ds = ds.rename_columns({"sentence": "text"})
    elif "content" in ds.column_names and "title" in ds.column_names:
        ds = ds.map(lambda x: {"text": x["title"] + " " + x["content"]})
    elif "content" in ds.column_names:
        ds = ds.rename_columns({"content": "text"})
    return ds


def standardize_label_column(ds):
    if "label" in ds.column_names and isinstance(ds.features["label"], ClassLabel):
        return ds.cast_column("label", Value("int64"))
    return ds


def add_label_text_column(ds, label_names):
    return ds.map(lambda x: {"label_text": label_names[x["label"]]})


def main():
    parser = argparse.ArgumentParser(
        description="Load two Hugging Face datasets, sample N examples from each, merge into train/test splits."
    )
    parser.add_argument("--dataset1", type=str, required=True)
    parser.add_argument("--dataset2", type=str, required=True)
    parser.add_argument("--split1", type=str, default="train")
    parser.add_argument("--split2", type=str, default="train")
    parser.add_argument("--eval_split1", type=str, default="test")
    parser.add_argument("--eval_split2", type=str, default="test")

    parser.add_argument("--num_samples1", type=int, default=4800)
    parser.add_argument("--num_samples2", type=int, default=4800)
    parser.add_argument("--eval_num_samples1", type=int, default=320)
    parser.add_argument("--eval_num_samples2", type=int, default=320)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--push_to_hub", action="store_true",
                        help="If set, push the final DatasetDict to Hugging Face Hub")
    parser.add_argument("--hub_dataset_name", type=str,
                        help="Required if --push_to_hub is set (e.g., 'your-username/merged_dataset')")

    args = parser.parse_args()

    ds1_train = load_dataset(args.dataset1, split=args.split1)
    ds2_train = load_dataset(args.dataset2, split=args.split2)
    ds1_eval  = load_dataset(args.dataset1, split=args.eval_split1)
    ds2_eval  = load_dataset(args.dataset2, split=args.eval_split2)

    label_names1 = ds1_train.features["label"].names if isinstance(ds1_train.features["label"], ClassLabel) else None
    label_names2 = ds2_train.features["label"].names if isinstance(ds2_train.features["label"], ClassLabel) else None

    ds1_train = add_label_text_column(standardize_label_column(normalize_columns(ds1_train)), label_names1)
    ds2_train = add_label_text_column(standardize_label_column(normalize_columns(ds2_train)), label_names2)
    ds1_eval  = add_label_text_column(standardize_label_column(normalize_columns(ds1_eval)),  label_names1)
    ds2_eval  = add_label_text_column(standardize_label_column(normalize_columns(ds2_eval)),  label_names2)

    ds1_train = ds1_train.shuffle(seed=args.seed).select(range(args.num_samples1))
    ds2_train = ds2_train.shuffle(seed=args.seed).select(range(args.num_samples2))
    ds1_eval  = ds1_eval.shuffle(seed=args.seed).select(range(args.eval_num_samples1))
    ds2_eval  = ds2_eval.shuffle(seed=args.seed).select(range(args.eval_num_samples2))

    merged_train = concatenate_datasets([ds1_train, ds2_train]).shuffle(seed=args.seed)
    merged_eval  = concatenate_datasets([ds1_eval, ds2_eval]).shuffle(seed=args.seed)

    final_ds = DatasetDict({
        "train": merged_train,
        "test": merged_eval
    })

    final_ds.save_to_disk(args.output_path)
    print(f"Merged dataset saved to {args.output_path}")
    print(f"Train size: {len(merged_train)}, Eval size: {len(merged_eval)}")

    if args.push_to_hub:
        if not args.hub_dataset_name:
            raise ValueError("Must provide --hub_dataset_name if --push_to_hub is set")
        final_ds.push_to_hub(args.hub_dataset_name)
        print(f"Pushed to Hugging Face Hub as: {args.hub_dataset_name}")


if __name__ == "__main__":
    main()