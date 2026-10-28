class ClassifierError(Exception):
    """Ошибка классификации, которую можно показать пользователю; status – HTTP-код ответа."""

    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status
