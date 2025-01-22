# computer_quirks

We have a lot of pet computers at work, and I'm tired of having to act differently on each one by hand.

I would prefer that some library (this one) would tell me the specifics of the computer I'm on, so I can act accordingly.

## install

```bash
conda activate whatever_conda_environment_you_want_to_install_this_into
cd ~/r
git clone git@github.com:darrenrhea/computer_quirks
cd computer_quirks
pip install -e . --no-deps
hash -r

what_computer_is_this

echo $(what_computer_is_this)
echo $(shared_dir)/baseball
echo $(what_os_is_this)
```

