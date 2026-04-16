import sys


class ProjectException(Exception):
    def __init__(self, error: Exception, context: str = ""):
        _, _, exc_tb = sys.exc_info()
        line = exc_tb.tb_lineno if exc_tb else "unknown"
        message = f"Error in {context} at line {line}: {error}"
        super().__init__(message)
