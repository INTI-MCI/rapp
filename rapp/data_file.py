import os


class DataFile:
    MESSAGE_OVERWRITE = "File already exists. Do you want to overwrite it? y/n (default is NO): "

    def __init__(self, overwrite=False, output_dir='data', prefix=None, delimiter='\t'):
        self._overwrite = overwrite
        self._output_dir = output_dir
        self._file = None
        self._prefix = prefix

        self.header = None
        self.column_names = None
        self.delimiter = delimiter

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
        row = "{}".format(self.delimiter).join(map(str, row)).expandtabs(10)
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
        if self.header is not None:
            self.write(self.header)

    def _add_column_names(self):
        if self.column_names is not None:
            names = "{}".format(self.delimiter).join(map(str, self.column_names)).expandtabs(10)
            self.write(names)
