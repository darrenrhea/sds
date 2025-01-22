class PasteableCutout(object):
    def __init__(self):
        pass
    def __repr__(self):
        return f"PasteableCutout({self.rgba_np_u8.shape}, {self.kind}, {self.file})"
