from typing import Dict, List


class Bounds(object):

    def __init__(
            self,
            k: List[float],
            skin: List[float],
            w: List[float],
            l: List[float],
            Pi: List[float],
            kv_kh: List[float],
            l_hor: List[float],
            xf: List[float],
            Cfd: List[float],
            n_frac: List[float],
    ):
        self.k = k
        self.skin = skin
        self.w = w
        self.l = l
        self.Pi = Pi
        self.kv_kh = kv_kh
        self.l_hor = l_hor
        self.xf = xf
        self.Cfd = Cfd
        self.n_frac = n_frac

    def __getstate__(self) -> Dict:
        state = {
            'k': self._convert_to_float(self.k),
            'skin': self._convert_to_float(self.skin),
            'w': self._convert_to_float(self.w),
            'l': self._convert_to_float(self.l),
            'Pi': self._convert_to_float(self.Pi),
            'kv_kh': self._convert_to_float(self.kv_kh),
            'l_hor': self._convert_to_float(self.l_hor),
            'xf': self._convert_to_float(self.xf),
            'Cfd': self._convert_to_float(self.Cfd),
            'n_frac': self._convert_to_float(self.n_frac),
        }
        return state

    @staticmethod
    def _convert_to_float(vector: List) -> List[float]:
        return [float(x) for x in vector]
