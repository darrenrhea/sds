# Distributed Hash Table

We store small files under their own sha256 hash digest,
usually expressed as a 32 digit lowercase hexidecimal string.

(
    Saying a file is "small" here effectively means "I can bear to wait for the calculation of its sha256 to happen."
)

So 300GB is probably not small, but if you are very patient then fine.


```bash
cd ~/r
git clone git@github.com:darrenrhea/distributed_hash_table
cd distributed_hash_table
pip install -e .
```

```
bat ~/r/assets_for_synthetic_data/baseball/backgrounds/busch4k.json
```
