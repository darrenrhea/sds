"""
This took the size of the checkpoints directory from 137GB to 9GB.
Not sure it is idempotent, so maybe dont run again.
"""
from colorama import Fore, Style
from pathlib import Path
from collections import defaultdict
import pprint as pp


checkpoints_dir = Path('/shared/checkpoints').resolve()

recipe_mapsto_epoch = defaultdict(list)

for p in checkpoints_dir.glob('*.pt'):
    maybe_six_digits = p.name[-9:-3]
    if all([digit.isdigit() for digit in maybe_six_digits]):
        is_number = True
    else:
        is_number = False
    if not is_number:
        continue

    epoch_str = p.name[-15:-9]
    print(epoch_str)
    if not epoch_str == "_epoch":
        continue
    recipe = p.name[:-15]
    
    print(f"{recipe=}")
    
    epoch_int = int(maybe_six_digits)

    recipe_mapsto_epoch[recipe].append(epoch_int)

for recipe, epochs in recipe_mapsto_epoch.items():
    recipe_mapsto_epoch[recipe] = sorted(epochs)


pp.pprint(recipe_mapsto_epoch)

for recipe, epochs in recipe_mapsto_epoch.items():
    epochs_to_destroy = epochs[:-1]
    keeper = epochs[-1]

    for epoch in epochs_to_destroy:
        to_destroy = checkpoints_dir / f"{recipe}_epoch{epoch:06d}.pt"
        print(f"{Fore.RED}Destroying {to_destroy}{Style.RESET_ALL}")
        # to_destroy.unlink()
    
    print(f"{Fore.GREEN}Keeping {recipe}_epoch{keeper:06d}.pt{Style.RESET_ALL}")

    

