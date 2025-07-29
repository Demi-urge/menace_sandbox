import os

if os.name == "nt":
    import msvcrt

    LOCK_EX = 0x1
    LOCK_SH = 0x2
    LOCK_NB = 0x4
    LOCK_UN = 0x8

    def flock(fd: int, flags: int) -> None:
        if flags & LOCK_UN:
            mode = msvcrt.LK_UNLCK
        elif flags & LOCK_EX:
            mode = msvcrt.LK_NBLCK if flags & LOCK_NB else msvcrt.LK_LOCK
        elif flags & LOCK_SH:
            mode = msvcrt.LK_NBRLCK if flags & LOCK_NB else msvcrt.LK_RLCK
        else:
            mode = msvcrt.LK_UNLCK
        msvcrt.locking(fd, mode, 1)
else:
    from fcntl import flock, LOCK_EX, LOCK_SH, LOCK_NB, LOCK_UN
