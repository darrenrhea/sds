from remote_directory_exists import remote_directory_exists

def test_remote_directory_exists_1():
    # if the directory exists, it should return True
    assert remote_directory_exists(
        remote_user="anna",
        remote_host="72.177.14.155",
        remote_ssh_port=41538,
        remote_directory_str="/home/anna/r/houston_rockets_book"
    )  == (True, True)


def test_remote_directory_exists_2():
    # if the directory does not exist, it should return False
    assert remote_directory_exists(
        remote_user="anna",
        remote_host="72.177.14.155",
        remote_ssh_port=41538,
        remote_directory_str="/home/anna/r/houston_rockets_book_abc"
    ) == (True, False)


def test_remote_directory_exists_3():
    assert remote_directory_exists(
        sshable_abbrev="appa",
        remote_directory_str="/home/anna/r/houston_rockets_book"
    )  == (True, True)


def test_remote_directory_exists_4():
    assert remote_directory_exists(
        sshable_abbrev="appa",
        remote_directory_str="/home/anna/r/houston_rockets_book_abc"
    )  == (True, False)


def test_remote_directory_exists_5(): 
    # if sshable_abbrev does not resolve, it should return (None, None)
    # since it doesn't know what host to connect to so it cannot help you.
    answer = remote_directory_exists(
        sshable_abbrev="djeee",
        remote_directory_str="/home/anna/r/houston_rockets_book_abc"
    )
    # print(f"{answer=}")
    assert answer == (None, None)

def test_remote_directory_exists_6():
    # files aren't directories, so this should return False
    assert remote_directory_exists(
        sshable_abbrev="appa",
        remote_directory_str="/home/anna/r/houston_rockets_book/doc/toc.rst"
    )  == (True, False)


def test_remote_directory_exists_7():
    # if it fails to connect, in this case because the port is
    # wrong, it should return (False, None)
    print("\nWarning: This test will take a while since the attempt to connect has to time out.")
    assert remote_directory_exists(
        remote_user="anna",
        remote_host="72.177.14.155",
        remote_ssh_port=12345,
        remote_directory_str="/home/anna/r/houston_rockets_book_abc"
    ) == (False, None)

if __name__ == "__main__":
    test_remote_directory_exists_1()
    test_remote_directory_exists_2()
    test_remote_directory_exists_3()
    test_remote_directory_exists_4()
    test_remote_directory_exists_5()
    test_remote_directory_exists_6()
    test_remote_directory_exists_7()
    print("remote_directory_exists_test.py has passed all tests")
    