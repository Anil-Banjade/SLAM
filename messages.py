from enum import StrEnum, auto 

class FrontendMessage(StrEnum):
    REQUEST_INIT=auto()
    ADD_KEYFRAME=auto()
    ADD_FRAME=auto()
    END=auto()


class BackendMessage(StrEnum):
    SYNC=auto()
    CHECKPOINT=auto()
    COMPLETED=auto()


