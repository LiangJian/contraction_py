import numpy as np

class Var(np.ndarray):

    def __new__(
        cls,
        name                    = "trk_pt",
        tree                    = None, # tree object
        eventNumber             = None,
        eventWeight             = None,
        numberOfBins            = None, # binning
        binningLogicSystem      = None, # binning
        shape                   = None,
        index                   = tuple(''),
        filename                = '',
        endian                  = '<'
        ):
        HeadType = np.dtype([
            ('head', [
                ('n_dims', 'i4'),
                ('one_dim', [
                    ('type', 'i4'),
                    ('n_indices', 'i4'),
                    ('indices', 'i4', 1024)
                ], 16)
            ])
        ])
        typename = ["other", "x", "y", "z", "t", "d", "c", "d2",
                    "c2", "complex", "mass", "smear", "displacement", "s_01", "s_02", "s_03",
                    "s_11", "s_12", "s_13", "d_01", "d_02", "d_03", "d_11", "d_12",
                    "d_13", "conf", "operator", "momentum", "direction", "t2", "mass2", "column",
                    "row", "temporary", "temporary2", "temporary3", "temporary4", "errorbar", "operator2", "param",
                    "fit_left", "fit_right", "jackknife", "jackknife2", "jackknife3", "jackknife4", "summary",
                    "channel",
                    "channel2", "eigen", "d_row", "d_col", "c_row", "c_col", "parity", "noise",
                    "evenodd", "disp_x", "disp_y", "disp_z", "disp_t", "t3", "t4", "t_source",
                    "t_current", "t_sink", "bootstrap", "nothing"]

        self = np.zeros(shape=shape).view(cls)

        if filename != '':
            head_data = np.fromfile(filename, dtype=HeadType, count=1)[0]
            n_dims = head_data['head']['n_dims']
            type = head_data['head']['one_dim']['type'][0:n_dims]
            typename = [typename[i] for i in type]
            n_indices = tuple(head_data['head']['one_dim']['n_indices'][0:n_dims])
            indices = head_data['head']['one_dim']['indices'][0:n_dims]
            DataType = np.dtype([
            ('head_tmp','S102400'),
            ('data',endian+'f8',n_indices)
            ])
            self = np.fromfile(filename, dtype=DataType)[0]['data']

        self._name              = name
        self.tree               = tree
        self.eventNumber        = eventNumber
        self.eventWeight        = eventWeight
        self.numberOfBins       = numberOfBins
        self.binningLogicSystem = binningLogicSystem
        self.index              = index

        return self

d = Var()
print(d.shape)
a = Var(shape=(2,2), index=('a', 'b'))
b = Var(shape=(2,2,2))
a = b
print(a.shape)

