class ModeError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class LossFunctionError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class OptionError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
