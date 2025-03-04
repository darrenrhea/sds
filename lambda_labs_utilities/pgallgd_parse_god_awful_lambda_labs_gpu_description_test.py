from print_green import (
     print_green
)
from pgallgd_parse_god_awful_lambda_labs_gpu_description import (
     pgallgd_parse_god_awful_lambda_labs_gpu_description
)


def test_pgallgd_parse_god_awful_lambda_labs_gpu_description_1():
    xys = [
        ("This computer has 128 GB of storage.", 128),
        ("The smartphone comes with 256 GB memory.", 256),
        ("No storage info here.", None),
        ("It has 512GB which is not separated by a space.", 512),
    ]
    
    for x, should_be in xys:
        result = pgallgd_parse_god_awful_lambda_labs_gpu_description(x)
        assert result == should_be
    
if __name__ == '__main__':
    test_pgallgd_parse_god_awful_lambda_labs_gpu_description_1()
    print_green("pgallgd_parse_god_awful_lambda_labs_gpu_description passed")