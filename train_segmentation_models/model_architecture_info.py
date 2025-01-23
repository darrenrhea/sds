"""
This is a place to store information about neural network architectures.
For one thing, the tests need a list of all possible archictectures so as to assert that
all the archs listed successfully train then infer on small, obvious segmentation problems.
Some models take in RGB, some BGR, some YUV444.
Some models want alexnet normalization, some just in [0.0, 1.0] i.e. divided by 255.
Some models need additional preprocessing before they can be used.
Some models have number-theoretic (like must be divisible by 32) constraints on the input width and height.
Some models take in a single batch tensor and return a single batch tensor,
but some return 2 tensors, and you need to know which one to use to derive labels, and how.
Some return a dictionary of tensors, and you need to know which one to use to derive labels and how.
Some models return 2 channels that need softmax-ing followed by taking channel 1 of that.
some return a single channel that needs sigmoiding.
"""


from typing import List


def get_valid_model_architecture_family_ids() -> List[str]:
    return [
        "effs",
        "effm",
        "effl",
        "duat",
        "res1",
        "res2",
        "effsp",
        "effsma",
        "ege",
        "replb",
        "repll",
        "evitb0",
        "evitb1",
        "evitb2",
        "evitb3",
        "u3effs",
        "u3effm",
        "u3res34",
        "u3res50",
        "u3res101",
        "u3convnexts",
        "u3convnextm",
        "u3fasternets",
        "u3fasternetm",
        "u3fasternetl",
        "resnet34basedunet",
    ]

# for people using this:
valid_model_architecture_ids = get_valid_model_architecture_family_ids()

first_coordinate_model_architecture_ids = [
    "effl",
    "effm",
    "effs",
    "effsma",
    "effsp",
]


second_coordinate_model_architecture_ids = [
    "ege",
]

# models that return a dictionary:
dict_model_architecture_ids = [
    "u3effs",
    "u3effm",
    "u3res34",
    "u3res50",
    "u3res101",
    "u3convnexts",
    "u3convnextm",
    "u3fasternets",
    "u3fasternetm",
    "u3fasternetl"
]


model_architecture_id_to_modulus_and_congruence_class = dict(
    effs=(32, 0),
    effm=(32, 0),
    effl=(32, 0),
)


def check_number_theory(
    model_architecture_id: str,
    patch_width: int,
    patch_height: int
):
    """
    Some models have number-theoretic constraints (like must be divisible by 32) on the input width and height.
    """
    assert isinstance(model_architecture_id, str)
    assert model_architecture_id in valid_model_architecture_ids
    assert isinstance(patch_width, int)
    assert isinstance(patch_height, int)
    if model_architecture_id in model_architecture_id_to_modulus_and_congruence_class:
        modulus, congruence_class = model_architecture_id_to_modulus_and_congruence_class[model_architecture_id]
        if patch_width % modulus != congruence_class:
            raise Exception(f"model {model_architecture_id} requires patch_width to be congruent to {congruence_class} mod {modulus}")
        if patch_height % modulus != congruence_class:
            raise Exception(f"model {model_architecture_id} requires patch_height to be congruent to {congruence_class} mod {modulus}")