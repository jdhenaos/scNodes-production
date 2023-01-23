import fire

from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO

import h5py
import json
import numpy as np
import pandas as pd
from matplotlib.image import imread
import anndata
from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,
)
from anndata import read as read_h5ad

###

import logging
import sys
from functools import update_wrapper, partial
from logging import CRITICAL, ERROR, WARNING, INFO, DEBUG
from datetime import datetime, timedelta, timezone
from typing import Optional, IO
import warnings

import anndata.logging


HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, 'HINT')

###

import inspect
import sys
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from time import time
from logging import getLevelName
from typing import Any, Union, Optional, Iterable, TextIO
from typing import Tuple, List, ContextManager

###

import sys
import inspect
from contextlib import contextmanager
import warnings
import importlib.util
from enum import Enum, IntEnum
from pathlib import Path
from weakref import WeakSet
from collections import namedtuple
from functools import partial, wraps
from types import ModuleType, MethodType
from typing import Union, Callable, Optional, Mapping, Any, Dict, Tuple, ContextManager

import numpy as np
from numpy import random
from scipy import sparse
from anndata import AnnData, __version__ as anndata_version
from textwrap import dedent
from packaging import version

#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#### LITERAL FUNCTION

try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass

#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#### ROOTLOGGER FUNCTION

HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, 'HINT')

class _RootLogger(logging.RootLogger):
    def __init__(self, level):
        super().__init__(level)
        self.propagate = False
        _RootLogger.manager = logging.Manager(self)

    def log(
        self,
        level: int,
        msg: str,
        *,
        extra: Optional[dict] = None,
        time: datetime = None,
        deep: Optional[str] = None,
    ) -> datetime:
        from . import settings

        now = datetime.now(timezone.utc)
        time_passed: timedelta = None if time is None else now - time
        extra = {
            **(extra or {}),
            'deep': deep if settings.verbosity.level < level else None,
            'time_passed': time_passed,
        }
        super().log(level, msg, extra=extra)
        return now

    def critical(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(CRITICAL, msg, time=time, deep=deep, extra=extra)

    def error(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(ERROR, msg, time=time, deep=deep, extra=extra)

    def warning(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(WARNING, msg, time=time, deep=deep, extra=extra)

    def info(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(INFO, msg, time=time, deep=deep, extra=extra)

    def hint(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(HINT, msg, time=time, deep=deep, extra=extra)

    def debug(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(DEBUG, msg, time=time, deep=deep, extra=extra)

def _set_log_file(settings):
    file = settings.logfile
    name = settings.logpath
    root = settings._root_logger
    h = logging.StreamHandler(file) if name is None else logging.FileHandler(name)
    h.setFormatter(_LogFormatter())
    h.setLevel(root.level)
    if len(root.handlers) == 1:
        root.removeHandler(root.handlers[0])
    elif len(root.handlers) > 1:
        raise RuntimeError('Scanpy’s root logger somehow got more than one handler')
    root.addHandler(h)


def _set_log_level(settings, level: int):
    root = settings._root_logger
    root.setLevel(level)
    (h,) = root.handlers  # may only be 1
    h.setLevel(level)


class _LogFormatter(logging.Formatter):
    def __init__(
        self, fmt='{levelname}: {message}', datefmt='%Y-%m-%d %H:%M', style='{'
    ):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord):
        format_orig = self._style._fmt
        if record.levelno == INFO:
            self._style._fmt = '{message}'
        elif record.levelno == HINT:
            self._style._fmt = '--> {message}'
        elif record.levelno == DEBUG:
            self._style._fmt = '    {message}'
        if record.time_passed:
            # strip microseconds
            if record.time_passed.microseconds:
                record.time_passed = timedelta(
                    seconds=int(record.time_passed.total_seconds())
                )
            if '{time_passed}' in record.msg:
                record.msg = record.msg.replace(
                    '{time_passed}', str(record.time_passed)
                )
            else:
                self._style._fmt += ' ({time_passed})'
        if record.deep:
            record.msg = f'{record.msg}: {record.deep}'
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


print_memory_usage = anndata.logging.print_memory_usage
get_memory_usage = anndata.logging.get_memory_usage


_DEPENDENCIES_NUMERICS = [
    'anndata',  # anndata actually shouldn't, but as long as it's in development
    'umap',
    'numpy',
    'scipy',
    'pandas',
    ('sklearn', 'scikit-learn'),
    'statsmodels',
    ('igraph', 'python-igraph'),
    'louvain',
    'leidenalg',
    'pynndescent',
]


def _versions_dependencies(dependencies):
    # this is not the same as the requirements!
    for mod in dependencies:
        mod_name, dist_name = mod if isinstance(mod, tuple) else (mod, mod)
        try:
            imp = __import__(mod_name)
            yield dist_name, imp.__version__
        except (ImportError, AttributeError):
            pass


def print_header(*, file=None):
    """\
    Versions that might influence the numerical results.
    Matplotlib and Seaborn are excluded from this.
    """

    modules = ['scanpy'] + _DEPENDENCIES_NUMERICS
    print(
        ' '.join(f'{mod}=={ver}' for mod, ver in _versions_dependencies(modules)),
        file=file or sys.stdout,
    )


def print_versions(*, file: Optional[IO[str]] = None):
    """\
    Print versions of imported packages, OS, and jupyter environment.

    For more options (including rich output) use `session_info.show` directly.
    """
    import session_info

    if file is not None:
        from contextlib import redirect_stdout

        warnings.warn(
            "Passing argument 'file' to print_versions is deprecated, and will be "
            "removed in a future version.",
            FutureWarning,
        )
        with redirect_stdout(file):
            print_versions()
    else:
        session_info.show(
            dependencies=True,
            html=False,
            excludes=[
                'builtins',
                'stdlib_list',
                'importlib_metadata',
                # Special module present if test coverage being calculated
                # https://gitlab.com/joelostblom/session_info/-/issues/10
                "$coverage",
            ],
        )


def print_version_and_date(*, file=None):
    """\
    Useful for starting a notebook so you see when you started working.
    """
    from . import __version__

    if file is None:
        file = sys.stdout
    print(
        f'Running Scanpy {__version__}, ' f'on {datetime.now():%Y-%m-%d %H:%M}.',
        file=file,
    )


def _copy_docs_and_signature(fn):
    return partial(update_wrapper, wrapped=fn, assigned=['__doc__', '__annotations__'])


def error(
    msg: str,
    *,
    time: datetime = None,
    deep: Optional[str] = None,
    extra: Optional[dict] = None,
) -> datetime:
    """\
    Log message with specific level and return current time.

    Parameters
    ==========
    msg
        Message to display.
    time
        A time in the past. If this is passed, the time difference from then
        to now is appended to `msg` as ` (HH:MM:SS)`.
        If `msg` contains `{time_passed}`, the time difference is instead
        inserted at that position.
    deep
        If the current verbosity is higher than the log function’s level,
        this gets displayed as well
    extra
        Additional values you can specify in `msg` like `{time_passed}`.
    """
    from ._settings import settings

    return settings._root_logger.error(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def warning(msg, *, time=None, deep=None, extra=None) -> datetime:
    from ._settings import settings

    return settings._root_logger.warning(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def info(msg, *, time=None, deep=None, extra=None) -> datetime:
    from ._settings import settings

    return settings._root_logger.info(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def hint(msg, *, time=None, deep=None, extra=None) -> datetime:
    from ._settings import settings

    return settings._root_logger.hint(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def debug(msg, *, time=None, deep=None, extra=None) -> datetime:
    from ._settings import settings

    return settings._root_logger.debug(msg, time=time, deep=deep, extra=extra)


#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#### SETTINGS FUNCTION

_VERBOSITY_TO_LOGLEVEL = {
    'error': 'ERROR',
    'warning': 'WARNING',
    'info': 'INFO',
    'hint': 'HINT',
    'debug': 'DEBUG',
}

for v, level in enumerate(list(_VERBOSITY_TO_LOGLEVEL.values())):
    _VERBOSITY_TO_LOGLEVEL[v] = level
class Verbosity(IntEnum):
    error = 0
    warn = 1
    info = 2
    hint = 3
    debug = 4

    @property
    def level(self) -> int:
        # getLevelName(str) returns the int level…
        return getLevelName(_VERBOSITY_TO_LOGLEVEL[self])

    @contextmanager
    def override(self, verbosity: "Verbosity") -> ContextManager["Verbosity"]:
        """\
        Temporarily override verbosity
        """
        settings.verbosity = verbosity
        yield self
        settings.verbosity = self


def _type_check(var: Any, varname: str, types: Union[type, Tuple[type, ...]]):
    if isinstance(var, types):
        return
    if isinstance(types, type):
        possible_types_str = types.__name__
    else:
        type_names = [t.__name__ for t in types]
        possible_types_str = "{} or {}".format(
            ", ".join(type_names[:-1]), type_names[-1]
        )
    raise TypeError(f"{varname} must be of type {possible_types_str}")


class ScanpyConfig:
    """\
    Config manager for scanpy.
    """

    def __init__(
        self,
        *,
        verbosity: str = "warning",
        plot_suffix: str = "",
        file_format_data: str = "h5ad",
        file_format_figs: str = "pdf",
        autosave: bool = False,
        autoshow: bool = True,
        writedir: Union[str, Path] = "./write/",
        cachedir: Union[str, Path] = "./cache/",
        datasetdir: Union[str, Path] = "./data/",
        figdir: Union[str, Path] = "./figures/",
        cache_compression: Union[str, None] = 'lzf',
        max_memory=15,
        n_jobs=1,
        logfile: Union[str, Path, None] = None,
        categories_to_ignore: Iterable[str] = ("N/A", "dontknow", "no_gate", "?"),
        _frameon: bool = True,
        _vector_friendly: bool = False,
        _low_resolution_warning: bool = True,
        n_pcs=50,
    ):
        # logging
        self._root_logger = _RootLogger(logging.INFO)  # level will be replaced
        self.logfile = logfile
        self.verbosity = verbosity
        # rest
        self.plot_suffix = plot_suffix
        self.file_format_data = file_format_data
        self.file_format_figs = file_format_figs
        self.autosave = autosave
        self.autoshow = autoshow
        self.writedir = writedir
        self.cachedir = cachedir
        self.datasetdir = datasetdir
        self.figdir = figdir
        self.cache_compression = cache_compression
        self.max_memory = max_memory
        self.n_jobs = n_jobs
        self.categories_to_ignore = categories_to_ignore
        self._frameon = _frameon
        """bool: See set_figure_params."""

        self._vector_friendly = _vector_friendly
        """Set to true if you want to include pngs in svgs and pdfs."""

        self._low_resolution_warning = _low_resolution_warning
        """Print warning when saving a figure with low resolution."""

        self._start = time()
        """Time when the settings module is first imported."""

        self._previous_time = self._start
        """Variable for timing program parts."""

        self._previous_memory_usage = -1
        """Stores the previous memory usage."""

        self.N_PCS = n_pcs
        """Default number of principal components to use."""

    @property
    def verbosity(self) -> Verbosity:
        """
        Verbosity level (default `warning`)

        Level 0: only show 'error' messages.
        Level 1: also show 'warning' messages.
        Level 2: also show 'info' messages.
        Level 3: also show 'hint' messages.
        Level 4: also show very detailed progress for 'debug'ging.
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity: Union[Verbosity, int, str]):
        verbosity_str_options = [
            v for v in _VERBOSITY_TO_LOGLEVEL if isinstance(v, str)
        ]
        if isinstance(verbosity, Verbosity):
            self._verbosity = verbosity
        elif isinstance(verbosity, int):
            self._verbosity = Verbosity(verbosity)
        elif isinstance(verbosity, str):
            verbosity = verbosity.lower()
            if verbosity not in verbosity_str_options:
                raise ValueError(
                    f"Cannot set verbosity to {verbosity}. "
                    f"Accepted string values are: {verbosity_str_options}"
                )
            else:
                self._verbosity = Verbosity(verbosity_str_options.index(verbosity))
        else:
            _type_check(verbosity, "verbosity", (str, int))
        _set_log_level(self, _VERBOSITY_TO_LOGLEVEL[self._verbosity])

    @property
    def plot_suffix(self) -> str:
        """Global suffix that is appended to figure filenames."""
        return self._plot_suffix

    @plot_suffix.setter
    def plot_suffix(self, plot_suffix: str):
        _type_check(plot_suffix, "plot_suffix", str)
        self._plot_suffix = plot_suffix

    @property
    def file_format_data(self) -> str:
        """File format for saving AnnData objects.

        Allowed are 'txt', 'csv' (comma separated value file) for exporting and 'h5ad'
        (hdf5) for lossless saving.
        """
        return self._file_format_data

    @file_format_data.setter
    def file_format_data(self, file_format: str):
        _type_check(file_format, "file_format_data", str)
        file_format_options = {"txt", "csv", "h5ad"}
        if file_format not in file_format_options:
            raise ValueError(
                f"Cannot set file_format_data to {file_format}. "
                f"Must be one of {file_format_options}"
            )
        self._file_format_data = file_format

    @property
    def file_format_figs(self) -> str:
        """File format for saving figures.

        For example 'png', 'pdf' or 'svg'. Many other formats work as well (see
        `matplotlib.pyplot.savefig`).
        """
        return self._file_format_figs

    @file_format_figs.setter
    def file_format_figs(self, figure_format: str):
        _type_check(figure_format, "figure_format_data", str)
        self._file_format_figs = figure_format

    @property
    def autosave(self) -> bool:
        """\
        Automatically save figures in :attr:`~scanpy._settings.ScanpyConfig.figdir` (default `False`).

        Do not show plots/figures interactively.
        """
        return self._autosave

    @autosave.setter
    def autosave(self, autosave: bool):
        _type_check(autosave, "autosave", bool)
        self._autosave = autosave

    @property
    def autoshow(self) -> bool:
        """\
        Automatically show figures if `autosave == False` (default `True`).

        There is no need to call the matplotlib pl.show() in this case.
        """
        return self._autoshow

    @autoshow.setter
    def autoshow(self, autoshow: bool):
        _type_check(autoshow, "autoshow", bool)
        self._autoshow = autoshow

    @property
    def writedir(self) -> Path:
        """\
        Directory where the function scanpy.write writes to by default.
        """
        return self._writedir

    @writedir.setter
    def writedir(self, writedir: Union[str, Path]):
        _type_check(writedir, "writedir", (str, Path))
        self._writedir = Path(writedir)

    @property
    def cachedir(self) -> Path:
        """\
        Directory for cache files (default `'./cache/'`).
        """
        return self._cachedir

    @cachedir.setter
    def cachedir(self, cachedir: Union[str, Path]):
        _type_check(cachedir, "cachedir", (str, Path))
        self._cachedir = Path(cachedir)

    @property
    def datasetdir(self) -> Path:
        """\
        Directory for example :mod:`~scanpy.datasets` (default `'./data/'`).
        """
        return self._datasetdir

    @datasetdir.setter
    def datasetdir(self, datasetdir: Union[str, Path]):
        _type_check(datasetdir, "datasetdir", (str, Path))
        self._datasetdir = Path(datasetdir).resolve()

    @property
    def figdir(self) -> Path:
        """\
        Directory for saving figures (default `'./figures/'`).
        """
        return self._figdir

    @figdir.setter
    def figdir(self, figdir: Union[str, Path]):
        _type_check(figdir, "figdir", (str, Path))
        self._figdir = Path(figdir)

    @property
    def cache_compression(self) -> Optional[str]:
        """\
        Compression for `sc.read(..., cache=True)` (default `'lzf'`).

        May be `'lzf'`, `'gzip'`, or `None`.
        """
        return self._cache_compression

    @cache_compression.setter
    def cache_compression(self, cache_compression: Optional[str]):
        if cache_compression not in {'lzf', 'gzip', None}:
            raise ValueError(
                f"`cache_compression` ({cache_compression}) "
                "must be in {'lzf', 'gzip', None}"
            )
        self._cache_compression = cache_compression

    @property
    def max_memory(self) -> Union[int, float]:
        """\
        Maximal memory usage in Gigabyte.

        Is currently not well respected....
        """
        return self._max_memory

    @max_memory.setter
    def max_memory(self, max_memory: Union[int, float]):
        _type_check(max_memory, "max_memory", (int, float))
        self._max_memory = max_memory

    @property
    def n_jobs(self) -> int:
        """\
        Default number of jobs/ CPUs to use for parallel computing.
        """
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        _type_check(n_jobs, "n_jobs", int)
        self._n_jobs = n_jobs

    @property
    def logpath(self) -> Optional[Path]:
        """\
        The file path `logfile` was set to.
        """
        return self._logpath

    @logpath.setter
    def logpath(self, logpath: Union[str, Path, None]):
        _type_check(logpath, "logfile", (str, Path))
        # set via “file object” branch of logfile.setter
        self.logfile = Path(logpath).open('a')
        self._logpath = Path(logpath)

    @property
    def logfile(self) -> TextIO:
        """\
        The open file to write logs to.

        Set it to a :class:`~pathlib.Path` or :class:`str` to open a new one.
        The default `None` corresponds to :obj:`sys.stdout` in jupyter notebooks
        and to :obj:`sys.stderr` otherwise.

        For backwards compatibility, setting it to `''` behaves like setting it to `None`.
        """
        return self._logfile

    @logfile.setter
    def logfile(self, logfile: Union[str, Path, TextIO, None]):
        if not hasattr(logfile, 'write') and logfile:
            self.logpath = logfile
        else:  # file object
            if not logfile:  # None or ''
                logfile = sys.stdout if self._is_run_from_ipython() else sys.stderr
            self._logfile = logfile
            self._logpath = None
            _set_log_file(self)

    @property
    def categories_to_ignore(self) -> List[str]:
        """\
        Categories that are omitted in plotting etc.
        """
        return self._categories_to_ignore

    @categories_to_ignore.setter
    def categories_to_ignore(self, categories_to_ignore: Iterable[str]):
        categories_to_ignore = list(categories_to_ignore)
        for i, cat in enumerate(categories_to_ignore):
            _type_check(cat, f"categories_to_ignore[{i}]", str)
        self._categories_to_ignore = categories_to_ignore

    # --------------------------------------------------------------------------------
    # Functions
    # --------------------------------------------------------------------------------

    # Collected from the print_* functions in matplotlib.backends
    # fmt: off
    _Format = Literal[
        'png', 'jpg', 'tif', 'tiff',
        'pdf', 'ps', 'eps', 'svg', 'svgz', 'pgf',
        'raw', 'rgba',
    ]
    # fmt: on

    def set_figure_params(
        self,
        scanpy: bool = True,
        dpi: int = 80,
        dpi_save: int = 150,
        frameon: bool = True,
        vector_friendly: bool = True,
        fontsize: int = 14,
        figsize: Optional[int] = None,
        color_map: Optional[str] = None,
        format: _Format = "pdf",
        facecolor: Optional[str] = None,
        transparent: bool = False,
        ipython_format: str = "png2x",
    ):
        """\
        Set resolution/size, styling and format of figures.

        Parameters
        ----------
        scanpy
            Init default values for :obj:`matplotlib.rcParams` suited for Scanpy.
        dpi
            Resolution of rendered figures – this influences the size of figures in notebooks.
        dpi_save
            Resolution of saved figures. This should typically be higher to achieve
            publication quality.
        frameon
            Add frames and axes labels to scatter plots.
        vector_friendly
            Plot scatter plots using `png` backend even when exporting as `pdf` or `svg`.
        fontsize
            Set the fontsize for several `rcParams` entries. Ignored if `scanpy=False`.
        figsize
            Set plt.rcParams['figure.figsize'].
        color_map
            Convenience method for setting the default color map. Ignored if `scanpy=False`.
        format
            This sets the default format for saving figures: `file_format_figs`.
        facecolor
            Sets backgrounds via `rcParams['figure.facecolor'] = facecolor` and
            `rcParams['axes.facecolor'] = facecolor`.
        transparent
            Save figures with transparent back ground. Sets
            `rcParams['savefig.transparent']`.
        ipython_format
            Only concerns the notebook/IPython environment; see
            :func:`~IPython.display.set_matplotlib_formats` for details.
        """
        if self._is_run_from_ipython():
            import IPython

            if isinstance(ipython_format, str):
                ipython_format = [ipython_format]
            IPython.display.set_matplotlib_formats(*ipython_format)

        from matplotlib import rcParams

        self._vector_friendly = vector_friendly
        self.file_format_figs = format
        if dpi is not None:
            rcParams["figure.dpi"] = dpi
        if dpi_save is not None:
            rcParams["savefig.dpi"] = dpi_save
        if transparent is not None:
            rcParams["savefig.transparent"] = transparent
        if facecolor is not None:
            rcParams['figure.facecolor'] = facecolor
            rcParams['axes.facecolor'] = facecolor
        if scanpy:
            from .plotting._rcmod import set_rcParams_scanpy

            set_rcParams_scanpy(fontsize=fontsize, color_map=color_map)
        if figsize is not None:
            rcParams['figure.figsize'] = figsize
        self._frameon = frameon

    @staticmethod
    def _is_run_from_ipython():
        """Determines whether we're currently in IPython."""
        import builtins

        return getattr(builtins, "__IPYTHON__", False)

    def __str__(self) -> str:
        return '\n'.join(
            f'{k} = {v!r}'
            for k, v in inspect.getmembers(self)
            if not k.startswith("_") and not k == 'getdoc'
        )


settings = ScanpyConfig()

#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#### EMPTY FUNCTION

class Empty(Enum):
    token = 0


_empty = Empty.token

#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#### READ FUNCTION

# .gz and .bz2 suffixes are also allowed for text formats
text_exts = {
    'csv',
    'tsv',
    'tab',
    'data',
    'txt',  # these four are all equivalent
}
avail_exts = {
    'anndata',
    'xlsx',
    'h5',
    'h5ad',
    'mtx',
    'mtx.gz',
    'soft.gz',
    'loom',
} | text_exts
"""Available file formats for reading data. """


# --------------------------------------------------------------------------------
# Reading and Writing data files and AnnData objects
# --------------------------------------------------------------------------------


def read(
    filename: Union[Path, str],
    backed: Optional[Literal['r', 'r+']] = None,
    sheet: Optional[str] = None,
    ext: Optional[str] = None,
    delimiter: Optional[str] = None,
    first_column_names: bool = False,
    backup_url: Optional[str] = None,
    cache: bool = False,
    cache_compression: Union[Literal['gzip', 'lzf'], None, Empty] = _empty,
    **kwargs,
) -> AnnData:
    """\
    Read file and return :class:`~anndata.AnnData` object.

    To speed up reading, consider passing ``cache=True``, which creates an hdf5
    cache file.

    Parameters
    ----------
    filename
        If the filename has no file extension, it is interpreted as a key for
        generating a filename via ``sc.settings.writedir / (filename +
        sc.settings.file_format_data)``.  This is the same behavior as in
        ``sc.read(filename, ...)``.
    backed
        If ``'r'``, load :class:`~anndata.AnnData` in ``backed`` mode instead
        of fully loading it into memory (`memory` mode). If you want to modify
        backed attributes of the AnnData object, you need to choose ``'r+'``.
    sheet
        Name of sheet/table in hdf5 or Excel file.
    ext
        Extension that indicates the file type. If ``None``, uses extension of
        filename.
    delimiter
        Delimiter that separates data within text file. If ``None``, will split at
        arbitrary number of white spaces, which is different from enforcing
        splitting at any single white space ``' '``.
    first_column_names
        Assume the first column stores row names. This is only necessary if
        these are not strings: strings in the first column are automatically
        assumed to be row names.
    backup_url
        Retrieve the file from an URL if not present on disk.
    cache
        If `False`, read from source, if `True`, read from fast 'h5ad' cache.
    cache_compression
        See the h5py :ref:`dataset_compression`.
        (Default: `settings.cache_compression`)
    kwargs
        Parameters passed to :func:`~anndata.read_loom`.

    Returns
    -------
    An :class:`~anndata.AnnData` object
    """
    filename = Path(filename)  # allow passing strings
    if is_valid_filename(filename):
        return _read(
            filename,
            backed=backed,
            sheet=sheet,
            ext=ext,
            delimiter=delimiter,
            first_column_names=first_column_names,
            backup_url=backup_url,
            cache=cache,
            cache_compression=cache_compression,
            **kwargs,
        )
    # generate filename and read to dict
    filekey = str(filename)
    filename = settings.writedir / (filekey + '.' + settings.file_format_data)
    if not filename.exists():
        raise ValueError(
            f'Reading with filekey {filekey!r} failed, '
            f'the inferred filename {filename!r} does not exist. '
            'If you intended to provide a filename, either use a filename '
            f'ending on one of the available extensions {avail_exts} '
            'or pass the parameter `ext`.'
        )
    return read_h5ad(filename, backed=backed)


def read_10x_h5(
    filename: Union[str, Path],
    genome: Optional[str] = None,
    gex_only: bool = True,
    backup_url: Optional[str] = None,
) -> AnnData:
    """\
    Read 10x-Genomics-formatted hdf5 file.

    Parameters
    ----------
    filename
        Path to a 10x hdf5 file.
    genome
        Filter expression to genes within this genome. For legacy 10x h5
        files, this must be provided if the data contains more than one genome.
    gex_only
        Only keep 'Gene Expression' data and ignore other feature types,
        e.g. 'Antibody Capture', 'CRISPR Guide Capture', or 'Custom'
    backup_url
        Retrieve the file from an URL if not present on disk.

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:

    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    """
    start = logg.info(f'reading {filename}')
    is_present = _check_datafile_present_and_download(filename, backup_url=backup_url)
    if not is_present:
        logg.debug(f'... did not find original file {filename}')
    with h5py.File(str(filename), 'r') as f:
        v3 = '/matrix' in f
    if v3:
        adata = _read_v3_10x_h5(filename, start=start)
        if genome:
            if genome not in adata.var['genome'].values:
                raise ValueError(
                    f"Could not find data corresponding to genome '{genome}' in '{filename}'. "
                    f'Available genomes are: {list(adata.var["genome"].unique())}.'
                )
            adata = adata[:, adata.var['genome'] == genome]
        if gex_only:
            adata = adata[:, adata.var['feature_types'] == 'Gene Expression']
        if adata.is_view:
            adata = adata.copy()
    else:
        adata = _read_legacy_10x_h5(filename, genome=genome, start=start)
    return adata


def _read_legacy_10x_h5(filename, *, genome=None, start=None):
    """
    Read hdf5 file from Cell Ranger v2 or earlier versions.
    """
    with h5py.File(str(filename), 'r') as f:
        try:
            children = list(f.keys())
            if not genome:
                if len(children) > 1:
                    raise ValueError(
                        f"'{filename}' contains more than one genome. For legacy 10x h5 "
                        "files you must specify the genome if more than one is present. "
                        f"Available genomes are: {children}"
                    )
                genome = children[0]
            elif genome not in children:
                raise ValueError(
                    f"Could not find genome '{genome}' in '{filename}'. "
                    f'Available genomes are: {children}'
                )

            dsets = {}
            _collect_datasets(dsets, f[genome])

            # AnnData works with csr matrices
            # 10x stores the transposed data, so we do the transposition right away
            from scipy.sparse import csr_matrix

            M, N = dsets['shape']
            data = dsets['data']
            if dsets['data'].dtype == np.dtype('int32'):
                data = dsets['data'].view('float32')
                data[:] = dsets['data']
            matrix = csr_matrix(
                (data, dsets['indices'], dsets['indptr']),
                shape=(N, M),
            )
            # the csc matrix is automatically the transposed csr matrix
            # as scanpy expects it, so, no need for a further transpostion
            adata = AnnData(
                matrix,
                obs=dict(obs_names=dsets['barcodes'].astype(str)),
                var=dict(
                    var_names=dsets['gene_names'].astype(str),
                    gene_ids=dsets['genes'].astype(str),
                ),
            )
            logg.info('', time=start)
            return adata
        except KeyError:
            raise Exception('File is missing one or more required datasets.')


def _collect_datasets(dsets: dict, group: h5py.Group):
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            dsets[k] = v[()]
        else:
            _collect_datasets(dsets, v)


def _read_v3_10x_h5(filename, *, start=None):
    """
    Read hdf5 file from Cell Ranger v3 or later versions.
    """
    with h5py.File(str(filename), 'r') as f:
        try:
            dsets = {}
            _collect_datasets(dsets, f["matrix"])

            from scipy.sparse import csr_matrix

            M, N = dsets['shape']
            data = dsets['data']
            if dsets['data'].dtype == np.dtype('int32'):
                data = dsets['data'].view('float32')
                data[:] = dsets['data']
            matrix = csr_matrix(
                (data, dsets['indices'], dsets['indptr']),
                shape=(N, M),
            )
            adata = AnnData(
                matrix,
                obs=dict(obs_names=dsets['barcodes'].astype(str)),
                var=dict(
                    var_names=dsets['name'].astype(str),
                    gene_ids=dsets['id'].astype(str),
                    feature_types=dsets['feature_type'].astype(str),
                    genome=dsets['genome'].astype(str),
                ),
            )
            logg.info('', time=start)
            return adata
        except KeyError:
            raise Exception('File is missing one or more required datasets.')


def read_visium(
    path: Union[str, Path],
    genome: Optional[str] = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str = None,
    load_images: Optional[bool] = True,
    source_image_path: Optional[Union[str, Path]] = None,
) -> AnnData:
    """\
    Read 10x-Genomics-formatted visum dataset.

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.

    See :func:`~scanpy.pl.spatial` for a compatible plotting function.

    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    source_image_path
        Path to the high-resolution tissue image. Path will be included in
        `.uns["spatial"][library_id]["metadata"]["source_image_path"]`.

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:

    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version', 'source_image_path'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    path = Path(path)
    adata = read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    from h5py import File

    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        files = dict(
            tissue_positions_file=path / 'spatial/tissue_positions_list.csv',
            scalefactors_json_file=path / 'spatial/scalefactors_json.json',
            hires_image=path / 'spatial/tissue_hires_image.png',
            lowres_image=path / 'spatial/tissue_lowres_image.png',
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]['images'] = dict()
        for res in ['hires', 'lowres']:
            try:
                adata.uns["spatial"][library_id]['images'][res] = imread(
                    str(files[f'{res}_image'])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]['scalefactors'] = json.loads(
            files['scalefactors_json_file'].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates
        positions = pd.read_csv(files['tissue_positions_file'], header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm['spatial'] = adata.obs[
            ['pxl_row_in_fullres', 'pxl_col_in_fullres']
        ].to_numpy()
        adata.obs.drop(
            columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata


def read_10x_mtx(
    path: Union[Path, str],
    var_names: Literal['gene_symbols', 'gene_ids'] = 'gene_symbols',
    make_unique: bool = True,
    cache: bool = False,
    cache_compression: Union[Literal['gzip', 'lzf'], None, Empty] = _empty,
    gex_only: bool = True,
    *,
    prefix: str = None,
) -> AnnData:
    """\
    Read 10x-Genomics-formatted mtx directory.

    Parameters
    ----------
    path
        Path to directory for `.mtx` and `.tsv` files,
        e.g. './filtered_gene_bc_matrices/hg19/'.
    var_names
        The variables index.
    make_unique
        Whether to make the variables index unique by appending '-1',
        '-2' etc. or not.
    cache
        If `False`, read from source, if `True`, read from fast 'h5ad' cache.
    cache_compression
        See the h5py :ref:`dataset_compression`.
        (Default: `settings.cache_compression`)
    gex_only
        Only keep 'Gene Expression' data and ignore other feature types,
        e.g. 'Antibody Capture', 'CRISPR Guide Capture', or 'Custom'
    prefix
        Any prefix before `matrix.mtx`, `genes.tsv` and `barcodes.tsv`. For instance,
        if the files are named `patientA_matrix.mtx`, `patientA_genes.tsv` and
        `patientA_barcodes.tsv` the prefix is `patientA_`.
        (Default: no prefix)

    Returns
    -------
    An :class:`~anndata.AnnData` object
    """
    path = Path(path)
    prefix = "" if prefix is None else prefix
    genefile_exists = (path / f'{prefix}genes.tsv').is_file()
    read = _read_legacy_10x_mtx if genefile_exists else _read_v3_10x_mtx
    adata = read(
        str(path),
        var_names=var_names,
        make_unique=make_unique,
        cache=cache,
        cache_compression=cache_compression,
        prefix=prefix,
    )
    if genefile_exists or not gex_only:
        return adata
    else:
        gex_rows = list(
            map(lambda x: x == 'Gene Expression', adata.var['feature_types'])
        )
        return adata[:, gex_rows].copy()


def _read_legacy_10x_mtx(
    path,
    var_names='gene_symbols',
    make_unique=True,
    cache=False,
    cache_compression=_empty,
    *,
    prefix="",
):
    """
    Read mex from output from Cell Ranger v2 or earlier versions
    """
    path = Path(path)
    adata = read(
        path / f'{prefix}matrix.mtx',
        cache=cache,
        cache_compression=cache_compression,
    ).T  # transpose the data
    genes = pd.read_csv(path / f'{prefix}genes.tsv', header=None, sep='\t')
    if var_names == 'gene_symbols':
        var_names = genes[1].values
        if make_unique:
            var_names = anndata.utils.make_index_unique(pd.Index(var_names))
        adata.var_names = var_names
        adata.var['gene_ids'] = genes[0].values
    elif var_names == 'gene_ids':
        adata.var_names = genes[0].values
        adata.var['gene_symbols'] = genes[1].values
    else:
        raise ValueError("`var_names` needs to be 'gene_symbols' or 'gene_ids'")
    adata.obs_names = pd.read_csv(path / f'{prefix}barcodes.tsv', header=None)[0].values
    return adata


def _read_v3_10x_mtx(
    path,
    var_names='gene_symbols',
    make_unique=True,
    cache=False,
    cache_compression=_empty,
    *,
    prefix="",
):
    """
    Read mtx from output from Cell Ranger v3 or later versions
    """
    path = Path(path)
    adata = read(
        path / f'{prefix}matrix.mtx.gz',
        cache=cache,
        cache_compression=cache_compression,
    ).T  # transpose the data
    genes = pd.read_csv(path / f'{prefix}features.tsv.gz', header=None, sep='\t')
    if var_names == 'gene_symbols':
        var_names = genes[1].values
        if make_unique:
            var_names = anndata.utils.make_index_unique(pd.Index(var_names))
        adata.var_names = var_names
        adata.var['gene_ids'] = genes[0].values
    elif var_names == 'gene_ids':
        adata.var_names = genes[0].values
        adata.var['gene_symbols'] = genes[1].values
    else:
        raise ValueError("`var_names` needs to be 'gene_symbols' or 'gene_ids'")
    adata.var['feature_types'] = genes[2].values
    adata.obs_names = pd.read_csv(path / f'{prefix}barcodes.tsv.gz', header=None)[
        0
    ].values
    return adata


def write(
    filename: Union[str, Path],
    adata: AnnData,
    ext: Optional[Literal['h5', 'csv', 'txt', 'npz']] = None,
    compression: Optional[Literal['gzip', 'lzf']] = 'gzip',
    compression_opts: Optional[int] = None,
):
    """\
    Write :class:`~anndata.AnnData` objects to file.

    Parameters
    ----------
    filename
        If the filename has no file extension, it is interpreted as a key for
        generating a filename via `sc.settings.writedir / (filename +
        sc.settings.file_format_data)`. This is the same behavior as in
        :func:`~scanpy.read`.
    adata
        Annotated data matrix.
    ext
        File extension from wich to infer file format. If `None`, defaults to
        `sc.settings.file_format_data`.
    compression
        See http://docs.h5py.org/en/latest/high/dataset.html.
    compression_opts
        See http://docs.h5py.org/en/latest/high/dataset.html.
    """
    filename = Path(filename)  # allow passing strings
    if is_valid_filename(filename):
        filename = filename
        ext_ = is_valid_filename(filename, return_ext=True)
        if ext is None:
            ext = ext_
        elif ext != ext_:
            raise ValueError(
                'It suffices to provide the file type by '
                'providing a proper extension to the filename.'
                'One of "txt", "csv", "h5" or "npz".'
            )
    else:
        key = filename
        ext = settings.file_format_data if ext is None else ext
        filename = _get_filename_from_key(key, ext)
    if ext == 'csv':
        adata.write_csvs(filename)
    else:
        adata.write(
            filename, compression=compression, compression_opts=compression_opts
        )


# -------------------------------------------------------------------------------
# Reading and writing parameter files
# -------------------------------------------------------------------------------


def read_params(
    filename: Union[Path, str], asheader: bool = False
) -> Dict[str, Union[int, float, bool, str, None]]:
    """\
    Read parameter dictionary from text file.

    Assumes that parameters are specified in the format::

        par1 = value1
        par2 = value2

    Comments that start with '#' are allowed.

    Parameters
    ----------
    filename
        Filename of data file.
    asheader
        Read the dictionary from the header (comment section) of a file.

    Returns
    -------
    Dictionary that stores parameters.
    """
    filename = str(filename)  # allow passing pathlib.Path objects
    from collections import OrderedDict

    params = OrderedDict([])
    for line in open(filename):
        if '=' in line:
            if not asheader or line.startswith('#'):
                line = line[1:] if line.startswith('#') else line
                key, val = line.split('=')
                key = key.strip()
                val = val.strip()
                params[key] = convert_string(val)
    return params


def write_params(path: Union[Path, str], *args, **maps):
    """\
    Write parameters to file, so that it's readable by read_params.

    Uses INI file format.
    """
    path = Path(path)
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True)
    if len(args) == 1:
        maps[None] = args[0]
    with path.open('w') as f:
        for header, map in maps.items():
            if header is not None:
                f.write(f'[{header}]\n')
            for key, val in map.items():
                f.write(f'{key} = {val}\n')


# -------------------------------------------------------------------------------
# Reading and Writing data files
# -------------------------------------------------------------------------------


def _read(
    filename: Path,
    backed=None,
    sheet=None,
    ext=None,
    delimiter=None,
    first_column_names=None,
    backup_url=None,
    cache=False,
    cache_compression=None,
    suppress_cache_warning=False,
    **kwargs,
):
    if ext is not None and ext not in avail_exts:
        raise ValueError(
            'Please provide one of the available extensions.\n' f'{avail_exts}'
        )
    else:
        ext = is_valid_filename(filename, return_ext=True)
    is_present = _check_datafile_present_and_download(filename, backup_url=backup_url)
    if not is_present:
        logg.debug(f'... did not find original file {filename}')
    # read hdf5 files
    if ext in {'h5', 'h5ad'}:
        if sheet is None:
            return read_h5ad(filename, backed=backed)
        else:
            logg.debug(f'reading sheet {sheet} from file {filename}')
            return read_hdf(filename, sheet)
    # read other file types
    path_cache = settings.cachedir / _slugify(filename).replace(
        '.' + ext, '.h5ad'
    )  # type: Path
    if path_cache.suffix in {'.gz', '.bz2'}:
        path_cache = path_cache.with_suffix('')
    if cache and path_cache.is_file():
        logg.info(f'... reading from cache file {path_cache}')
        return read_h5ad(path_cache)

    if not is_present:
        raise FileNotFoundError(f'Did not find file {filename}.')
    logg.debug(f'reading {filename}')
    if not cache and not suppress_cache_warning:
        logg.hint(
            'This might be very slow. Consider passing `cache=True`, '
            'which enables much faster reading from a cache file.'
        )
    # do the actual reading
    if ext == 'xlsx' or ext == 'xls':
        if sheet is None:
            raise ValueError("Provide `sheet` parameter when reading '.xlsx' files.")
        else:
            adata = read_excel(filename, sheet)
    elif ext in {'mtx', 'mtx.gz'}:
        adata = read_mtx(filename)
    elif ext == 'csv':
        adata = read_csv(filename, first_column_names=first_column_names)
    elif ext in {'txt', 'tab', 'data', 'tsv'}:
        if ext == 'data':
            logg.hint(
                "... assuming '.data' means tab or white-space " 'separated text file',
            )
            logg.hint('change this by passing `ext` to sc.read')
        adata = read_text(filename, delimiter, first_column_names)
    elif ext == 'soft.gz':
        adata = _read_softgz(filename)
    elif ext == 'loom':
        adata = read_loom(filename=filename, **kwargs)
    else:
        raise ValueError(f'Unknown extension {ext}.')
    if cache:
        logg.info(
            f'... writing an {settings.file_format_data} '
            'cache file to speedup reading next time'
        )
        if cache_compression is _empty:
            cache_compression = settings.cache_compression
        if not path_cache.parent.is_dir():
            path_cache.parent.mkdir(parents=True)
        # write for faster reading when calling the next time
        adata.write(path_cache, compression=cache_compression)
    return adata


def _slugify(path: Union[str, PurePath]) -> str:
    """Make a path into a filename."""
    if not isinstance(path, PurePath):
        path = PurePath(path)
    parts = list(path.parts)
    if parts[0] == '/':
        parts.pop(0)
    elif len(parts[0]) == 3 and parts[0][1:] == ':\\':
        parts[0] = parts[0][0]  # C:\ → C
    filename = '-'.join(parts)
    assert '/' not in filename, filename
    assert not filename[1:].startswith(':'), filename
    return filename


def _read_softgz(filename: Union[str, bytes, Path, BinaryIO]) -> AnnData:
    """\
    Read a SOFT format data file.

    The SOFT format is documented here
    http://www.ncbi.nlm.nih.gov/geo/info/soft2.html.

    Notes
    -----
    The function is based on a script by Kerby Shedden.
    http://dept.stat.lsa.umich.edu/~kshedden/Python-Workshop/gene_expression_comparison.html
    """
    import gzip

    with gzip.open(filename, mode='rt') as file:
        # The header part of the file contains information about the
        # samples. Read that information first.
        samples_info = {}
        for line in file:
            if line.startswith("!dataset_table_begin"):
                break
            elif line.startswith("!subset_description"):
                subset_description = line.split("=")[1].strip()
            elif line.startswith("!subset_sample_id"):
                subset_ids = line.split("=")[1].split(",")
                subset_ids = [x.strip() for x in subset_ids]
                for k in subset_ids:
                    samples_info[k] = subset_description
        # Next line is the column headers (sample id's)
        sample_names = file.readline().strip().split("\t")
        # The column indices that contain gene expression data
        indices = [i for i, x in enumerate(sample_names) if x.startswith("GSM")]
        # Restrict the column headers to those that we keep
        sample_names = [sample_names[i] for i in indices]
        # Get a list of sample labels
        groups = [samples_info[k] for k in sample_names]
        # Read the gene expression data as a list of lists, also get the gene
        # identifiers
        gene_names, X = [], []
        for line in file:
            # This is what signals the end of the gene expression data
            # section in the file
            if line.startswith("!dataset_table_end"):
                break
            V = line.split("\t")
            # Extract the values that correspond to gene expression measures
            # and convert the strings to numbers
            x = [float(V[i]) for i in indices]
            X.append(x)
            gene_names.append(V[1])
    # Convert the Python list of lists to a Numpy array and transpose to match
    # the Scanpy convention of storing samples in rows and variables in colums.
    X = np.array(X).T
    obs = pd.DataFrame({"groups": groups}, index=sample_names)
    var = pd.DataFrame(index=gene_names)
    return AnnData(X=X, obs=obs, var=var, dtype=X.dtype)


# -------------------------------------------------------------------------------
# Type conversion
# -------------------------------------------------------------------------------


def is_float(string: str) -> float:
    """Check whether string is float.

    See also
    --------
    http://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_int(string: str) -> bool:
    """Check whether string is integer."""
    try:
        int(string)
        return True
    except ValueError:
        return False


def convert_bool(string: str) -> Tuple[bool, bool]:
    """Check whether string is boolean."""
    if string == 'True':
        return True, True
    elif string == 'False':
        return True, False
    else:
        return False, False


def convert_string(string: str) -> Union[int, float, bool, str, None]:
    """Convert string to int, float or bool."""
    if is_int(string):
        return int(string)
    elif is_float(string):
        return float(string)
    elif convert_bool(string)[0]:
        return convert_bool(string)[1]
    elif string == 'None':
        return None
    else:
        return string


# -------------------------------------------------------------------------------
# Helper functions for reading and writing
# -------------------------------------------------------------------------------


def get_used_files():
    """Get files used by processes with name scanpy."""
    import psutil

    loop_over_scanpy_processes = (
        proc for proc in psutil.process_iter() if proc.name() == 'scanpy'
    )
    filenames = []
    for proc in loop_over_scanpy_processes:
        try:
            flist = proc.open_files()
            for nt in flist:
                filenames.append(nt.path)
        # This catches a race condition where a process ends
        # before we can examine its files
        except psutil.NoSuchProcess:
            pass
    return set(filenames)


def _get_filename_from_key(key, ext=None) -> Path:
    ext = settings.file_format_data if ext is None else ext
    return settings.writedir / f'{key}.{ext}'


def _download(url: str, path: Path):
    try:
        import ipywidgets
        from tqdm.auto import tqdm
    except ImportError:
        from tqdm import tqdm

    from urllib.request import urlopen, Request
    from urllib.error import URLError

    blocksize = 1024 * 8
    blocknum = 0

    try:
        req = Request(url, headers={"User-agent": "scanpy-user"})

        try:
            open_url = urlopen(req)
        except URLError:
            logging.warning(
                'Failed to open the url with default certificates, trying with certifi.'
            )

            from certifi import where
            from ssl import create_default_context

            open_url = urlopen(req, context=create_default_context(cafile=where()))

        with open_url as resp:
            total = resp.info().get("content-length", None)
            with tqdm(
                unit="B",
                unit_scale=True,
                miniters=1,
                unit_divisor=1024,
                total=total if total is None else int(total),
            ) as t, path.open("wb") as f:
                block = resp.read(blocksize)
                while block:
                    f.write(block)
                    blocknum += 1
                    t.update(len(block))
                    block = resp.read(blocksize)

    except (KeyboardInterrupt, Exception):
        # Make sure file doesn’t exist half-downloaded
        if path.is_file():
            path.unlink()
        raise


def _check_datafile_present_and_download(path, backup_url=None):
    """Check whether the file is present, otherwise download."""
    path = Path(path)
    if path.is_file():
        return True
    if backup_url is None:
        return False
    logging.info(
        f'try downloading from url\n{backup_url}\n'
        '... this may take a while but only happens once'
    )
    if not path.parent.is_dir():
        logg.info(f'creating directory {path.parent}/ for saving data')
        path.parent.mkdir(parents=True)

    _download(backup_url, path)
    return True


def is_valid_filename(filename: Path, return_ext=False):
    """Check whether the argument is a filename."""
    ext = filename.suffixes

    if len(ext) > 2:
        logg.warning(
            f'Your filename has more than two extensions: {ext}.\n'
            f'Only considering the two last: {ext[-2:]}.'
        )
        ext = ext[-2:]

    # cases for gzipped/bzipped text files
    if len(ext) == 2 and ext[0][1:] in text_exts and ext[1][1:] in ('gz', 'bz2'):
        return ext[0][1:] if return_ext else True
    elif ext and ext[-1][1:] in avail_exts:
        return ext[-1][1:] if return_ext else True
    elif ''.join(ext) == '.soft.gz':
        return 'soft.gz' if return_ext else True
    elif ''.join(ext) == '.mtx.gz':
        return 'mtx.gz' if return_ext else True
    elif not return_ext:
        return False
    raise ValueError(
        f'''\
{filename!r} does not end on a valid extension.
Please, provide one of the available extensions.
{avail_exts}
Text files with .gz and .bz2 extensions are also supported.\
'''
    )


#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#### MAIN FUNCTION

def pbmc3k(): #-> ad.AnnData:
    """\
    3k PBMCs from 10x Genomics.

    The data consists in 3k PBMCs from a Healthy Donor and is freely available
    from 10x Genomics (`here
    <http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz>`__
    from this `webpage
    <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k>`__).

    The exact same data is also used in Seurat's
    `basic clustering tutorial <https://satijalab.org/seurat/pbmc3k_tutorial.html>`__.

    .. note::

        This downloads 5.9 MB of data upon the first call of the function and stores it in `./data/pbmc3k_raw.h5ad`.

    The following code was run to produce the file.

    .. code:: python

        adata = sc.read_10x_mtx(
            # the directory with the `.mtx` file
            './data/filtered_gene_bc_matrices/hg19/',
            # use gene symbols for the variable names (variables-axis index)
            var_names='gene_symbols',
            # write a cache file for faster subsequent reading
            cache=True,
        )

        adata.var_names_make_unique()  # this is unnecessary if using 'gene_ids'
        adata.write('write/pbmc3k_raw.h5ad', compression='gzip')

    Returns
    -------
    Annotated data matrix.
    """
    url = 'http://falexwolf.de/data/pbmc3k_raw.h5ad'
    adata = read(settings.datasetdir / 'pbmc3k_raw.h5ad', backup_url=url)

    return "%s Dataset succesfully loaded!!" % url

    #return adata

if __name__ == '__main__':
    fire.Fire(pbmc3k)