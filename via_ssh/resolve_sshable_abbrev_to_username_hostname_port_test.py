import pprint
from resolve_sshable_abbrev_to_username_hostname_port import (
     resolve_sshable_abbrev_to_username_hostname_port
)


def test_resolve_sshable_abbrev_to_username_hostname_port_1():
    ans = resolve_sshable_abbrev_to_username_hostname_port(
        sshable_abbrev="global_appa"
    )
    # pprint.pprint(ans)
    assert ans == ('anna', '72.177.14.155', 41538)



def test_resolve_sshable_abbrev_to_username_hostname_port_2():
    ans = resolve_sshable_abbrev_to_username_hostname_port(
        remote_host="192.168.0.16",
        remote_user="anna",
        remote_ssh_port=41538
    )
    assert ans == ('anna', '192.168.0.16', 41538, )


def test_resolve_sshable_abbrev_to_username_hostname_port_3():
    ans = resolve_sshable_abbrev_to_username_hostname_port(
        sshable_abbrev="zeus",
    )
    # pprint.pprint(ans)
    assert ans == ('drhea', '45.56.121.210', 22, )


if __name__ == "__main__":
    test_resolve_sshable_abbrev_to_username_hostname_port_1()
    test_resolve_sshable_abbrev_to_username_hostname_port_2()
    test_resolve_sshable_abbrev_to_username_hostname_port_3()
    print("All tests of resolve_sshable_abbrev_to_username_hostname_port.py passed.")
    