import better_json as bj

def migrate_record(old_record):
    new_record = {}
    keys_to_keep = ["clip_id", "frame_index", "clip_id_info"]
    old_label_name_to_new_label_name = {
        "mask": "floor_not_floor",
        "depthmap": "depth_map"
    }
    label_name_to_sha256 = {}
    for key, value in old_record.items():
        if key in keys_to_keep:
            new_record[key] = old_record[key]
    for key, value in old_record.items():
        if key.endswith("_sha256"):
            old_label_name = key[:-7]
            if old_label_name in old_label_name_to_new_label_name:
                new_label_name = old_label_name_to_new_label_name[old_label_name]
            else:
                new_label_name = old_label_name
            label_name_to_sha256[new_label_name] = old_record[key]
    new_record["label_name_to_sha256"] = label_name_to_sha256
    return new_record  

def migrate():
    old_records= bj.load('better.json')

    new_records = []
    for old_record in old_records:
        new_record = migrate_record(old_record)
        new_records.append(new_record)

    bj.dump("new.json", new_records)

if __name__ == "__main__":
    migrate()
