from remote_file_exists import remote_file_exists

def test_remote_file_exists_1():
    # if the file exists, it should return True
    assert (
        remote_file_exists(
            remote_user="anna",
            remote_host="72.177.14.155",
            remote_ssh_port=41538,
            remote_abs_file_path_str="/home/anna/an_extant_file.txt"
        )  == (True, True)
    ), "remote_file_exists_test.py: test_remote_file_exists_1() failed"


def test_remote_file_exists_2():
    # if the file does not exist, it should return False
    assert remote_file_exists(
        remote_user="anna",
        remote_host="72.177.14.155",
        remote_ssh_port=41538,
        remote_abs_file_path_str="/home/anna/a_non_existent_file.txt"
    ) == (True, False)


def test_remote_file_exists_3():
    assert remote_file_exists(
        sshable_abbrev="appa",
        remote_abs_file_path_str="/home/anna/an_extant_file.txt"
    )  == (True, True)


def test_remote_file_exists_4():
    assert remote_file_exists(
        sshable_abbrev="appa",
        remote_abs_file_path_str="/home/anna/a_non_existent_file.txt"
    )  == (True, False)


def test_remote_file_exists_5(): 
    # if sshable_abbrev does not resolve, it should return (None, None)
    # since it doesn't know what host to connect to so it cannot help you.
    assert remote_file_exists(
        sshable_abbrev="djeee",
        remote_abs_file_path_str="/home/anna/whatever_it_will_fail_to_connect"
    )  == (None, None)

def test_remote_file_exists_6():
    # directories aren't files, so this should return False
    assert remote_file_exists(
        sshable_abbrev="appa",
        remote_abs_file_path_str="/home/anna/r"
    )  == (True, False)


def test_remote_file_exists_7():
    # if it fails to connect, in this case because the port is
    # wrong, it should return (False, None)
    assert remote_file_exists(
        remote_user="anna",
        remote_host="72.177.14.155",
        remote_ssh_port=12345,
        remote_abs_file_path_str="/home/anna/r/houston_rockets_book_abc"
    ) == (False, None)

if __name__ == "__main__":
    test_remote_file_exists_1()
    test_remote_file_exists_2()
    test_remote_file_exists_3()
    test_remote_file_exists_4()
    test_remote_file_exists_5()
    test_remote_file_exists_6()
    test_remote_file_exists_7()
    print("remote_file_exists_test.py has passed all tests")
    