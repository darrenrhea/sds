from chunk_list import chunk_list
import numpy as np

def test_chunk_list_1():
    length = np.random.randint(0, 100)
    my_list = list(range(length))
    num_chunks = np.random.randint(1, 99)
    chunks = chunk_list(
        lst=my_list,
        num_chunks=num_chunks
    )
    
    print(chunks)
    total = sum(chunks, [])
    assert my_list == total

if __name__ == "__main__":
    test_chunk_list_1()