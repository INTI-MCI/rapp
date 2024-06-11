import os


class DataFile:
    MESSAGE_OVERWRITE = "File already exists. Do you want to overwrite it? y/n (default is NO): "

    def __init__(
        self,
        overwrite: bool = False,
        output_dir: str = 'data',
        header: str = None,
        column_names: list = None,
        prefix: str = None,
        delimiter: str = ','
    ):
        self._overwrite = overwrite
        self._output_dir = output_dir
        self._header = header
        self._column_names = column_names
        self._prefix = prefix
        self._delimiter = delimiter

        self._file = None

    @property
    def path(self):
        if self._file is not None:
            return self._file.name

        return None

    def open(self, filename):
        os.makedirs(self._output_dir, exist_ok=True)
        if self._prefix is not None:
            filename = "{}-{}".format(self._prefix, filename)

        path = os.path.join(self._output_dir, filename)

        mode = 'a'
        if not os.path.exists(path) or self._do_overwrite():
            mode = 'w'

        self._file = open(path, mode=mode)

        if mode == 'w':
            self._add_header()
            self._add_column_names()

        return self._file

    def add_row(self, row):
        """Adds row to the file.

        Args:
            row: iterable with values of each column.
        """
        row = "{}".format(self._delimiter).join(map(str, row))
        self.write(row)

    def write(self, string, new_line=True):
        self._file.write(string)

        if new_line:
            self._file.write("\n")

    def close(self):
        self._file.close()

    def _do_overwrite(self):
        if self._overwrite or input(self.MESSAGE_OVERWRITE) == 'y':
            return True

        return False

    def _add_header(self):
        if self._header is not None:
            self.write(self._header)

    def _add_column_names(self):
        if self._column_names is not None:
            names = "{}".format(self._delimiter).join(map(str, self._column_names))
            self.write(names)

    def remove(self):
        self.close()
        os.remove(self._file.name)
