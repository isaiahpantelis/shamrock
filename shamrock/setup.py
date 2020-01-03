"""
At the command prompt: $ python setup.py build_ext --inplace
"""
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os

os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

# -- Create the right path for the source files.
cwd = os.path.dirname(__file__)

# sources = ['shamrock.pyx']
# sources = [os.path.join(cwd, _) for _ in sources]

# -- List of Extension objects
ext_modules = [
    Extension(
        name='shamrock_cy',
        sources=[os.path.join(cwd, _) for _ in ['shamrock_cy.pyx']],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O2", "-march=native", '-stdlib=libc++', '-std=c++17'],
        extra_link_args=["-O3", "-march=native", '-stdlib=libc++'],
        language="c++",
        include_dirs=[".", np.get_include()],
    )
]

for _ in ext_modules:
    _.compiler_directives = {'language_level': 3}

setup(
    ext_modules=ext_modules, cmdclass={'build_ext': build_ext}
)

# from Cython.Build import cythonize
# setup(ext_modules=cythonize(['shamrock/cheb.pyx']))
