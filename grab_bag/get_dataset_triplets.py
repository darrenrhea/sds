from pathlib import Path

def get_dataset_triplets(dataset_dir, dataset_extension):
    out_triplets = []
    out_triplet = ()
    for member in dataset_dir.rglob("*"):
        if member.is_file() and str(member).endswith(".jpg"):
            original_path = member
            parent_dir = member.parent
            member_name_prefix = member.stem
            mask_png = member_name_prefix + dataset_extension
            mask_path = parent_dir / mask_png
            if mask_path.exists():
                out_triplet = (original_path, mask_path, None)
                print(out_triplet) 
                out_triplets.append(out_triplet) 
    
    print(len(out_triplets))
    return out_triplets          

if __name__ == "__main__":

    dataset_dir = Path(f"/mnt/nas/volume1/videos/segmentation_datasets").expanduser()
    dataset_extension = "_nonfloor.png"

    get_dataset_triplets(dataset_dir=dataset_dir, dataset_extension=dataset_extension)