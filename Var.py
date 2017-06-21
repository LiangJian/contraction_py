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
        filename                = '',
        endian                  = '<',
        data                    = None
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

        if data is not None:
            self = np.array(data).view(cls)

        if filename != '':
            head_data_ = np.fromfile(filename, dtype=HeadType, count=1)[0]
            n_dims_ = head_data_['head']['n_dims']
            type_ = head_data_['head']['one_dim']['type'][0:n_dims_]
            typename_ = [typename[i] for i in type_]
            n_indices_ = tuple(head_data_['head']['one_dim']['n_indices'][0:n_dims_])
            indices_ = head_data_['head']['one_dim']['indices'][0:n_dims_]
            DataType_ = np.dtype([
            ('head_tmp','S102400'),
            ('data',endian+'f8',n_indices_)
            ])
            self = np.fromfile(filename, dtype=DataType_)[0]['data'].view(cls)
            self.indices = {}
            for i in range(n_dims_):
                self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
            self.type = typename_
            self.index = {}
            for i in range(n_dims_):
                self.index[typename_[i]] = i


        self._name              = name
        self.tree               = tree
        self.eventNumber        = eventNumber
        self.eventWeight        = eventWeight
        self.numberOfBins       = numberOfBins
        self.binningLogicSystem = binningLogicSystem
        return self

    def find_name(self, name_=''):
        if name_ in self.type:
            return self.index[name_]
        else:
            return -1

    def jack(self):
        index_ = self.find_name("conf")
        if index_ == -1:
            print("no index conf")
            exit(-1)
        #return jack(self,index_)

d = Var()
print(d.shape)
a = Var(shape=(2,2))
b = Var(shape=(2,2,2))
a = b
print(a.shape)

a = Var(filename="rbc_conf_2464_m0.005_0.04_000495_hyp.2pt.dov.glue.data")
print(a.shape)
print(a.indices['momentum'])
print(a.indices[a.type[a.index['momentum']]])
print(a.type)
print(a.index['momentum'])
print(a.find_name(name_='momentum'))
print(a[0,0,0,0,1,0,0])
print(a.flat[0])
print(a.flat[-1])