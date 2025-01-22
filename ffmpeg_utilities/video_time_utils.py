import numpy as np


def int_to_hour_minute_second_frag_quad(frame_index, fps=59.94):
    """
    In light of avidemux, this is wrong. 
    We should convert from frame_index to hh:mm:ss.milliseconds
    """
    seconds = frame_index / fps
    hour = int(np.floor(seconds / 3600))
    assert 0 <= hour and hour <= 4
    seconds -= 3600 * hour
    minute = int(np.floor(seconds / 60))
    assert 0 <= minute and minute <= 59
    seconds -= 60 * minute
    second = int(np.floor(seconds))
    assert 0 <= second and second <= 59
    seconds -= second
    fragment = int(np.floor(seconds * 60))
    assert 0 <= fragment and fragment <= 59
    return (hour, minute, second, fragment)


def int_to_hours_minutes_seconds_milliseconds_quadruplet(frame_index, fps=59.94):
    """
    In light of avidemux,
    Convert from frame_index to hh:mm:ss.<three digits of milliseconds>
    """
    seconds = frame_index / fps
    hour = int(np.floor(seconds / 3600))
    assert 0 <= hour and hour <= 4
    seconds -= 3600 * hour
    minute = int(np.floor(seconds / 60))
    assert 0 <= minute and minute <= 59
    seconds -= 60 * minute
    second = int(np.floor(seconds))
    assert 0 <= second and second <= 59
    seconds -= second
    fragment = int(np.round(seconds * 1000))
    assert 0 <= fragment and fragment <= 999
    return (hour, minute, second, fragment)


def hhmmss_to_hours_minutes_seconds_milliseconds_quadrulet(
    hhmmss: str
):
    """
    Given a timecode string like those seen in avidemux,
    like ``00:00:38.433`` return a triple of ints like (0, 0, 38, 433)
    """
    assert len(hhmmss) == 8 or len(hhmmss) == 12
    for i in [0, 1, 3, 4, 6, 7]:
        assert hhmmss[i] in "0123456789"
    assert hhmmss[2] == ":"
    assert hhmmss[5] == ":"
    hh, mm, ssdotfrag = hhmmss.split(":")
    if len(hhmmss) > 8:
        ss, frag = ssdotfrag.split(".")
    else:
        ss = ssdotfrag
        frag = "000"
    hour = int(hh)
    assert 0 <= hour and hour <= 4
    minute = int(mm)
    assert 0 <= minute and minute <= 59
    second = int(ss)
    assert 0 <= second and second <= 59
    millisecond = int(frag)
    assert 0 <= millisecond and millisecond <= 999
    return (hour, minute, second, millisecond)


def hhmmss_to_int(hhmmss, fps=59.94):
    (hour, minute, second, milliseconds) = \
        hhmmss_to_hours_minutes_seconds_milliseconds_quadrulet(hhmmss)
    seconds = 3600 * hour + 60 * minute + second + milliseconds / 1000
    frame_index = int(round(fps * seconds))
    return frame_index


def int_to_hhmmss(frame_index, fps=59.94):
    (hour, minute, second, fragment) = int_to_hour_minute_second_frag_quad(
       frame_index=frame_index,
       fps=fps
    )
    hhmmss = f"{hour:02d}:{minute:02d}:{second:02d}"
    return hhmmss

def test_int_to_hhmmss():
    assert int_to_hhmmss(164236) == "00:45:40", f"ERROR: {int_to_hhmmss(164236) }"

    assert int_to_hours_minutes_seconds_milliseconds_quadruplet(164236) == (0, 45, 40, 0)

def test_hhmmss_to_hours_minutes_seconds_milliseconds_quadrulet():
    hhmmss = "00:00:38.433"
    assert hhmmss_to_hours_minutes_seconds_milliseconds_quadrulet(hhmmss) == (0, 0, 38, 433)
    hhmmss = "01:02:38"
    assert hhmmss_to_hours_minutes_seconds_milliseconds_quadrulet(hhmmss) == (1, 2, 38, 0)


if __name__ == "__main__":
    test_int_to_hhmmss()
    test_hhmmss_to_hours_minutes_seconds_milliseconds_quadrulet()
    print("Everything passed")