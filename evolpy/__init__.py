num_digits = 3

def v_print(self, *a, **k):
        text = print(*a, **k)
        if text is not None:
            tqdm.write(text)

from evolpy.ga import GA
from evolpy.gp import GP

__all__ = ['GA', 'GP']