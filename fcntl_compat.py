import os

if os.name == "nt":
    import msvcrt
    import errno

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

    def ioctl(fd: int, op: int, arg: int = 0, mutate_flag: bool = True) -> int:
        """Basic ``ioctl`` compatibility wrapper for Windows.

        The sandbox only relies on ``ioctl`` for file locking behaviour on
        Unix platforms. Windows does not offer a direct equivalent so we
        approximate the locking semantics using :mod:`msvcrt` and simply
        perform a no-op for unsupported operations.  The function returns ``0``
        to mimic the ``fcntl.ioctl`` return value.

        Parameters mirror :func:`fcntl.ioctl` but are largely ignored on
        Windows. Unsupported operations raise :class:`OSError` with
        ``errno.ENOSYS``.
        """

        if op in (LOCK_EX, LOCK_SH):
            # ``arg`` is unused; we always lock a single byte similar to
            # ``flock`` above.
            mode = msvcrt.LK_LOCK if op == LOCK_EX else msvcrt.LK_RLCK
            msvcrt.locking(fd, mode, 1)
            return 0
        if op == LOCK_UN:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return 0
        raise OSError(errno.ENOSYS, "ioctl operation not supported on Windows")
else:
    from fcntl import flock, ioctl, LOCK_EX, LOCK_SH, LOCK_NB, LOCK_UN
