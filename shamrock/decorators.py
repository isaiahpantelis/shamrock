from typing import Callable

import shamrock as sh
from shamrock import shamrock_cy
from shamrock.common import errmsg


def ChebProxy(I=(-1.0, 1.0), K=None, settings=shamrock_cy.Settings_cy()):

    print('-- Inside `ChebProxy`')

    def decorator(f):
        print('-- Inside `decorator`')
        return sh.ChebProxy(f, I, K, settings)

    return decorator