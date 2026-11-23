"""Đánh dấu deprecated cho gói `sna.metrics`; bộ đánh giá chính thức là `sna.evaluation`."""
import functools
import warnings


def deprecated_metric(replacement):
    """Phát DeprecationWarning mỗi lần gọi, không đổi giá trị trả về."""
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f'{func.__module__}.{func.__name__} đã deprecated; dùng {replacement} '
                '(giao thức đánh giá cuối kỳ nằm ở sna.evaluation).',
                DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorate
